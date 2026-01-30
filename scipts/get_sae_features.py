#!/usr/bin/env python3

import os
import gc
import argparse
from typing import List, Tuple

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from sae_lens import SAE, HookedSAETransformer


def extract_sae_activations(
    texts: List[str],
    classes: List[str],
    model,
    sae,
    tokenizer,
    checkpoint_dir: str,
    checkpoint_prefix: str,
    checkpoint_interval: int = 100,
    start_idx: int = 0
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    model.eval()
    os.makedirs(checkpoint_dir, exist_ok=True)

    # we are using 4 different types of aggregations
    all_cont_activations = []
    all_any_activations = []
    all_last_token_activations = []
    all_binary_sum_activations = []
    processed_texts = []
    processed_classes = []

    checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_prefix}_checkpoint.npz")
    resume_from = start_idx

    if os.path.exists(checkpoint_path):
        print(f"Found existing checkpoint at {checkpoint_path}")
        checkpoint_data = np.load(checkpoint_path, allow_pickle=True)

        all_cont_activations = checkpoint_data['cont_activations'].tolist()
        all_any_activations = checkpoint_data['any_activations'].tolist()
        all_last_token_activations = checkpoint_data['last_token_activations'].tolist()
        all_binary_sum_activations = checkpoint_data['binary_sum_activations'].tolist()
        processed_texts = checkpoint_data['texts'].tolist()
        processed_classes = checkpoint_data['classes'].tolist()
        resume_from = checkpoint_data['last_processed_idx'].item() + 1

        print(f"Resuming from index {resume_from}")
    else:
        print(f"No checkpoint found. Starting from index {start_idx}")
        resume_from = start_idx

    # cannot use batching due to limited VRAM
    for i in tqdm(
        range(resume_from, len(texts)), 
        desc="Extracting SAE activations", 
        initial=resume_from, 
        total=len(texts)
        ):
        text = texts[i]
        cls = classes[i]
        
        formatted_text = tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True
        )

        tokens = model.to_tokens(formatted_text, prepend_bos=False)
        
        try:
            with torch.inference_mode():
                _, cache = model.run_with_cache_with_saes(tokens, saes=[sae])
                sae_acts_cpu = cache[f"{sae.cfg.metadata.hook_name}.hook_sae_acts_post"].cpu()
                del cache
                del tokens
                torch.cuda.empty_cache()

                start_pos = 4 # it has been shown it's best to avoid special tokens
                end_pos = sae_acts_cpu.shape[1]

                cont_activation = sae_acts_cpu[0, start_pos:end_pos, :].sum(dim=0).numpy()
                any_token_activation = ((sae_acts_cpu[0, start_pos:end_pos, :] > 0).sum(dim=0) > 0).numpy()
                last_token_activation = sae_acts_cpu[0, end_pos - 6, :].numpy()
                binary_sum_activation = (sae_acts_cpu[0, start_pos:end_pos, :] > 0).sum(dim=0).numpy()

                del sae_acts_cpu

            all_cont_activations.append(cont_activation)
            all_any_activations.append(any_token_activation)
            all_last_token_activations.append(last_token_activation)
            all_binary_sum_activations.append(binary_sum_activation)
            processed_texts.append(text)
            processed_classes.append(cls)

        except Exception as e:
            print(f"Error processing index {i}: {e}")
            continue

        if i % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        if (i + 1) % checkpoint_interval == 0:
            np.savez_compressed(
                checkpoint_path,
                cont_activations=np.array(all_cont_activations),
                any_activations=np.array(all_any_activations),
                last_token_activations=np.array(all_last_token_activations),
                binary_sum_activations=np.array(all_binary_sum_activations),
                texts=np.array(processed_texts, dtype=object),
                classes=np.array(processed_classes, dtype=object),
                last_processed_idx=np.array(i)
            )

    np.savez_compressed(
        checkpoint_path,
        cont_activations=np.array(all_cont_activations),
        any_activations=np.array(all_any_activations),
        last_token_activations=np.array(all_last_token_activations),
        binary_sum_activations=np.array(all_binary_sum_activations),
        texts=np.array(processed_texts, dtype=object),
        classes=np.array(processed_classes, dtype=object),
        last_processed_idx=np.array(len(texts) - 1)
    )

    return (
        processed_texts,
        processed_classes,
        np.array(all_cont_activations),
        np.array(all_any_activations),
        np.array(all_last_token_activations),
        np.array(all_binary_sum_activations)
    )


def load_and_prepare_data(
    data_path: str,
    char_threshold: int = 4000, # to avoid OOMs - VRAM is limited
    sample_size: int = 20000 # 20k per each of the two classes
) -> pd.DataFrame:

    df = pd.read_csv(data_path)
    df = df[["text", "class"]]
    df = df[df['text'].str.len() <= char_threshold]
    
    df_depression = df[df["class"] == "depression"]
    df_depression = df_depression.sample(n=sample_size, random_state=42)
    
    df_suicide = df[df["class"] == "SuicideWatch"]
    df_suicide = df_suicide.sample(n=sample_size, random_state=42)
    df_suicide['class'] = 'suicide'
    
    df_combined = pd.concat([df_depression, df_suicide], ignore_index=True)
    return df_combined


def initialize_model_and_sae(
    model_name: str,
    layer: int,
    sae_width: str,
    device: str
) -> Tuple:

    model = HookedSAETransformer.from_pretrained(model_name, device=device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sae = SAE.from_pretrained(
        release="gemma-scope-9b-it-res-canonical",
        sae_id=f"layer_{layer}/width_{sae_width}/canonical",
        device=device,
    )
    return model, sae, tokenizer


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract SAE activations from text data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the input CSV file"
    )
    parser.add_argument(
        "--char_threshold",
        type=int,
        default=4000,
        help="Maximum character length for texts"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=20000,
        help="Number of samples per class"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-2-9b-it",
        choices=["google/gemma-2-9b-it"], # need to apply to access this model through HF
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=31,
        choices=list(range(42)), # TO-DO: set correct choices
        help="Layer number for SAE extraction"
    )
    parser.add_argument(
        "--sae_width",
        type=str,
        default="16k",
        choices=["16k", "131k"], # only two options are available publicly
        help="SAE width (16k or 131k)"
    )
    
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Directory for saving checkpoints"
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=300,
        help="Save checkpoint every N iterations"
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        default="depr_suic_20k.npz",
        help="Output file path for final combined data"
    )
    
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token for authentication"
    )
    
    return parser.parse_args()


def main():
    args = parse_arguments()

    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)
    
    df = load_and_prepare_data(
        data_path=args.data_path,
        char_threshold=args.char_threshold,
        sample_size=args.sample_size
    )
    
    model, sae, tokenizer = initialize_model_and_sae(
        model_name=args.model_name,
        layer=args.layer,
        sae_width=args.sae_width,
        device=device
    )
    
    checkpoint_prefix = f"depr_suic_{args.model_name.replace('/', '_')}_layer{args.layer}_width{args.sae_width}"
    
    texts, classes, cont_activations, any_activations, last_token_activations, binary_sum_activations = extract_sae_activations(
        texts=df['text'].tolist(),
        classes=df['class'].tolist(),
        model=model,
        sae=sae,
        tokenizer=tokenizer,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_prefix=checkpoint_prefix,
        checkpoint_interval=args.checkpoint_interval
    )
    
    print(f"Final shape: {cont_activations.shape}")
    print(f"Classes distribution: {np.unique(classes, return_counts=True)}")
    
    np.savez_compressed(
        args.output_file,
        cont_activations=cont_activations,
        any_activations=any_activations,
        last_token_activations=last_token_activations,
        binary_sum_activations=binary_sum_activations,
        classes=classes,
        texts=texts
    )


if __name__ == "__main__":
    main()