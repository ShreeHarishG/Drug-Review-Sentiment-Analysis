# Drug Review Sentiment Analysis — Capstone Project
> **Blackbelt Data Science Capstone Project | Analytics Vidhya**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Project Overview

This capstone project addresses **Drug-Conditioned Sentiment Classification**, predicting patient sentiment towards specific drugs mentioned within review texts. The dataset presents a multi-class NLP task with three target classes:
- `0`: **Positive Sentiment**
- `1`: **Negative Sentiment**
- `2`: **Neutral Sentiment**

### The Core Challenge: Drug-Conditioned NLP
A single patient review may discuss multiple drugs simultaneously (e.g., comparing side effects or switching treatments). The sentiment label is **drug-specific** — evaluating the target drug rather than the general tone of the full comment.

To prevent misclassification, models must be explicitly conditioned on the target drug for every instance.

---

## 🏗️ Methodology & Technical Highlights

```
                       Raw Dataset (train.csv, test.csv)
                                       │
                      Exploratory Data Analysis (EDA)
                Class Imbalance | Text Length | Drug Distribution
                                       │
                      Drug-Conditioned Preprocessing
                 [Target Drug Name] + [Patient Review Text]
                                       │
                    Stratified Train / Validation Split (80/20)
                                       │
        ┌──────────────────────────────┴──────────────────────────────┐
        │                                                             │
 classical TF-IDF Pipeline                                 Transformer Pipeline
 (Unigrams + Bigrams)                                     (Smart Head-Tail Truncation)
        │                                                             │
 Baseline / Logistic Regression / NB / SVM                    Fine-Tuned DeBERTa-v3
        │                                                             │
        └──────────────────────────────┬──────────────────────────────┘
                                       │
                        Model Comparison (Weighted F1)
                                       │
                        Final Prediction & Submission
```

### Key Technical Strategies
1. **Drug-Conditioned Text Formatting**: Target drug names are prepended directly to the review text (`"Drug: <Name> | Review: <Text>"`). This prevents identical reviews from receiving unconditioned predictions when paired with different drugs.
2. **Dual Preprocessing Pipelines**:
   - *Classical Models (TF-IDF)*: Heavy text normalization (lowercasing, HTML stripping, punctuation removal).
   - *Transformer (DeBERTa-v3)*: Minimal preprocessing preserving casing, punctuation, and sub-word structural cues.
3. **Smart Head-Tail Truncation**: To fit transformer token limits without losing critical context, long reviews preserve tokens from both the **beginning** (drug mention) and **end** (concluding sentiment), dropping the middle.
4. **Class-Imbalance Strategy**: Standard cross-entropy loss is modified using class-frequency weighting (`class_weight="balanced"`) rather than synthetic oversampling, ensuring robust performance on minority classes.

---

## 📊 Model Evaluation & Results

All models were evaluated on the exact same stratified 80/20 validation split. The primary evaluation metric is **Weighted F1-Score** (as specified by competition rules), tracked alongside **Macro F1** and **Accuracy**.

| Rank | Model | Accuracy | Weighted F1 ⭐ | Macro F1 | Status |
|:---:|:---|:---:|:---:|:---:|:---|
| 🥇 | **DeBERTa-v3 (Fine-tuned)** | **0.6913** | **0.7102** | **0.6198** | ✅ **Selected Final Model** |
| 🥈 | **Linear SVM (Tuned)** | 0.7045 | 0.6888 | 0.4943 | Strong Classical Baseline |
| 🥉 | **Linear SVM (Untuned)** | 0.7045 | 0.6888 | 0.4943 | Baseline Comparison |
| 4 | **Logistic Regression** | 0.6316 | 0.6514 | 0.5001 | Classical Baseline |
| 5 | **Baseline (Majority Class)** | 0.7244 | 0.6087 | 0.2801 | Reference Benchmark |
| 6 | **Multinomial Naive Bayes** | 0.7225 | 0.6077 | 0.2796 | Classical Baseline |

### Key Findings
- **DeBERTa-v3-small** outperformed all classical baselines with a **0.7102 Weighted F1** and a significantly higher **Macro F1 (0.6198)**, demonstrating superior capability in recognizing minority sentiment classes.
- Linear SVM achieved the strongest score among classical approaches (**0.6888 Weighted F1**).

---

## 📁 Repository Structure

```
.
├── data/
│   ├── train.csv                      # Raw training dataset
│   ├── test.csv                       # Raw test dataset
│   └── sample_submission.csv          # Submission format template
├── docs/
│   ├── Blackbelt_Capstone.pdf         # Problem statement & capstone brief
│   └── Capstone_General_Instructions.pdf
├── notebooks/
│   ├── EDA_Baseline.ipynb             # Exploratory Data Analysis & initial baselines
│   ├── DeBERTa_v3_Training.ipynb      # Transformer fine-tuning pipeline
│   └── Drug_Sentiment_Analysis_Final.ipynb # Complete end-to-end Capstone notebook
├── output/
│   └── LinearSVM.csv                  # Baseline prediction output
├── .gitignore                         # Configured Git exclusions (excludes model weights >100MB)
└── README.md                          # Project documentation
```

---

## 🛠️ Getting Started & Local Setup

### 1. Prerequisites
- Python 3.10 or higher
- NVIDIA GPU with CUDA support (recommended for transformer fine-tuning)

### 2. Installation
Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/ShreeHarishG/Drug-Review-Sentiment-Analysis.git
cd Drug-Review-Sentiment-Analysis

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install torch transformers datasets scikit-learn pandas numpy lightgbm xgboost matplotlib seaborn
```

### 3. Running the Project
Open Jupyter Notebook or JupyterLab to execute the pipeline:

```bash
jupyter lab
```

Navigate to `notebooks/Drug_Sentiment_Analysis_Final.ipynb` and run the cells sequentially to reproduce the preprocessing, training, model evaluation, and final test prediction generation.

---

## 💡 Challenges & Engineering Solutions

- **GitHub 100 MB Limit**: Large binary model weights (`*.safetensors`, `*.pt`, `*.pkl`) were excluded from Git tracking using detailed `.gitignore` rules to prevent repository upload failures and RPC disconnects.
- **Cold-Start Drugs**: Prepending drug names directly into text inputs allowed sub-word tokenization to generalize effectively even for rare or unseen test drugs.

---

## 🔮 Future Improvements

- Explore larger transformer architectures (`microsoft/deberta-v3-base` or `deberta-v3-large`).
- Implement a two-tower neural network (dedicated drug encoder + text encoder).
- Apply SHAP/LIME explainability tools for interpretability of sentiment predictions.
- Experiment with ensembling DeBERTa-v3 and Linear SVM predictions via weighted voting.

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
