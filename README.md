# Development of Interpretable Suicide Risk Detection System from Social Media Posts

**HSE University, Faculty of Computer Science**  
**Master's Degree in Data Science**  
**Thesis Project**

---

### 👤 Student
**Arsenii Pliusnin** (Арсений Александрович Плюснин)

### 👨‍🏫 Scientific Supervisor
**Dmitry Ilvovsky** (Дмитрий Алексеевич Ильвовский)

---

## 📋 Project Overview

This research focuses on developing an interpretable machine learning system for detecting suicide risk indicators in social media (Reddit) posts. The project employs Sparse Autoencoders (SAE) to extract interpretable features from language model representations, combined with explainable AI techniques (SHAP values) to provide insights into model predictions.

---

## 🗓️ Work Plan

### **Phase 1: Literature Review and Theoretical Foundation** *(December 2025)*

- **1.1** Review of computational approaches to suicide risk detection
- **1.2** Analysis of available datasets and benchmarks in the domain ✅
- **1.3** Theoretical background:
  - Large Language Models (LLMs) and Transformer architectures
  - Contemporary Text Classification methods
  - Sparse Autoencoders (SAE) and Mechanistic Interpretability
  - SHAP values and explainable AI techniques

### **Phase 2: Data Collection and Preprocessing** *(December 2025)*

- **2.1** Dataset acquisition and exploratory data analysis (EDA) ✅
- **2.2** Data preprocessing and cleaning ✅
- **2.3** Train/validation/test split preparation ✅

### **Phase 3: Baseline Models Development** *(December 2025)*

- **3.1** Implementation of baseline classifiers: ✅
  - Logistic Regression over tf-idf representations ✅
- **3.2** DeBERTa-based classifier as strong baseline ✅
- **3.4** Baselines quality measurements ✅

### **Phase 4: SAE-based Classifier Development** *(January 2026)*

- **4.1** SAE feature extraction from language model representations ✅
- **4.2** Classifier architecture selection and development ✅

### **Phase 5: Model Comparison and Evaluation** *(January 2026)*

- **5.1** Comprehensive comparison of all classifiers on unified test set: ✅
  - SAE-based classifier ✅
  - DeBERTa baseline ✅
  - Traditional ML baselines (Logistic Regression, etc.) ✅

### **Phase 6: Interpretability Implementation** *(February 2026)*
- **6.1** SHAP explainer integration with SAE-based classifier ✅
- **6.2** SAE features interpretation using Neuronpedia: ✅
  - Automated feature description generation ✅
- **6.3** Individual prediction explanation visualization development ✅

### **Phase 7: Service Development** *(March 2026)*

- Development of a demonstration service that showcases the suicide risk detection system

### **Phase 8: Final Analysis and Preparations** *(March 2026)*

- **8.1** Comprehensive results analysis and discussion
- **8.2** Thesis writing

---

## 🎯 Key Research Questions

1. Can Sparse Autoencoder features provide competitive classification performance compared to end-to-end neural approaches?
2. How does the interpretability-performance trade-off manifest in SAE-based classifiers?
3. Which interpretability method provides the most actionable insights for suicide risk detection?
4. Can SAE features be meaningfully interpreted in the context of mental health assessment?

---

## 🛠️ Technologies and Tools

- **Language Models**: DeBERTa, SAE-equipped transformers (Gemma-2-instruction-tuned-9B)
- **ML Frameworks**: PyTorch, scikit-learn, LightGBM, Transformers, SAE_Lens
- **Interpretability**: SHAP, Neuronpedia

---

## 📚 References

*(To be populated during literature review phase)*

---

## 📧 Contact

For questions or collaboration inquiries, please contact:
- **Student**: Arsenii Pliusnin (arsenii.pliusnin@gmail.com)
- **Supervisor**: Dmitry Ilvovsky (dilvovsky@hse.ru)
