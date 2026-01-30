# Implicit Sexism Detection with Retrieval-Based Explanations

This project implements a complete system for **detecting implicit sexism in online text** (e.g., tweets, Reddit comments) and **explaining model predictions** using **retrieval-based justifications**.  
The work is inspired by the **SemEval EDOS (Explainable Detection of Online Sexism)** task and focuses on combining **classification accuracy** with **interpretability and user trust**.

---

## 1. Project Objectives

The goals of this project are:

- Detect **implicit sexism** in short online texts
- Provide **human-understandable explanations** for predictions
- Retrieve **similar annotated examples** and **definitions** as evidence
- Explore whether **data augmentation (DDA/CSE)** improves performance
- Evaluate both **classification accuracy** and **explanation quality**

---

## 2. Tasks Implemented

### Task 1 — Implicit Sexism Classification
- Binary classification: **Sexist** vs **Not Sexist**
- Uses **Sentence-BERT embeddings**
- Classifier: **Logistic Regression**
- Dataset: **EDOS aggregated dataset**
- Evaluation: accuracy, precision, recall, F1-score

### Task 2 — Retrieval-Based Explanation (RAG-style)
- Retrieves:
  - Semantically similar sexist examples
  - Definitions of implicit sexism
- Generates explanations grounded in retrieved evidence
- Does **not rely on large language models**
- Implements a lightweight **Retrieval-Augmented Generation (RAG)** approach

### Optional — DDA / CSE Experiments
- Definition-based Data Augmentation (DDA)
- Contextual Semantic Expansion (CSE)
- Included as experimental extensions
- Results reported even when performance decreases

---

## 3. System Architecture

Input Text
│
▼
Sentence-BERT Encoder
│
├──▶ Task 1: Logistic Regression Classifier
│ └── Prediction: Sexist / Not Sexist
│
└──▶ Task 2: Similarity Retriever
├── EDOS Example Sentences
└── Sexism Definitions
│
▼
Explanation Generator


---

## 4. Repository Structure

BSP_Implicit_Sexism/
│
├── data/
│ ├── raw/
│ │ └── edos_labelled_aggregated.csv
│ └── processed/
│ ├── task1_train.csv
│ └── task1_test.csv
│
├── src/
│ ├── data_utils.py # Data preparation for Task 1
│ ├── train_task1.py # Baseline classifier training
│ ├── train_task1_dda_cse.py # DDA / CSE experiment
│ ├── augmentation.py # DDA / CSE logic
│ ├── retrieval.py # Similarity-based retrieval
│ ├── explain.py # Explanation generation
│ └── run_task2_demo.py # End-to-end demo for Task 2
│
├── results/
│ ├── task1_baseline_metrics.txt
│ └── task1_dda_cse_metrics.txt
│
├── venv/ # Python virtual environment
└── README.md


---

## 5. Installation & Setup

### Step 1 — Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate    # Windows

### Step 2 — Install dependencies
pip install sentence-transformers scikit-learn pandas numpy torch

#6. How to Run the Project
Step 1 — Prepare the Dataset (Task 1)
python src/data_utils.py


This:

Loads the EDOS dataset

Creates train/test splits

Saves processed files to data/processed/

Step 2 — Train the Baseline Classifier (Task 1)
python src/train_task1.py


Outputs:

Classification report printed to terminal

Metrics saved to:

results/task1_baseline_metrics.txt

Optional — Run DDA / CSE Experiment
python src/train_task1_dda_cse.py


Purpose:

Explore whether augmentation improves implicit sexism detection

Compare against baseline performance

Results saved to:

results/task1_dda_cse_metrics.txt

Step 3 — Run Task 2 Explanation Demo
python src/run_task2_demo.py


Example output:

Input text: "She is too emotional to lead a team."
Prediction: Sexist

Explanation:
The sentence was classified as sexist because it aligns with known
patterns of implicit sexism, supported by the following retrieved
examples and definitions:

1. Retrieved example (similarity=0.82):
   "Women are naturally worse at math."

2. Retrieved definition:
   "Implicit sexism refers to subtle or indirect expressions of gender bias."

7. Evaluation
Task 1 — Classification Metrics

Accuracy

Precision

Recall

F1-score

Macro and weighted averages

Task 2 — Explanation Quality

Semantic relevance of retrieved items

Interpretability of explanations

Qualitative inspection of justifications

8. Dataset

EDOS (Explainable Detection of Online Sexism)

SemEval shared task dataset

Aggregated labels used for binary classification

9. Research Findings

Baseline classifier performs well on non-sexist content

Recall for implicit sexism remains challenging

DDA / CSE did not improve performance in this setup

Negative results are documented as part of the research contribution

10. Conclusion

This project delivers:

A working implicit sexism classifier

A retrieval-based explanation module

Quantitative and qualitative evaluation

Optional augmentation experiments

An interpretable and reproducible NLP pipeline

The system meets the requirements of the SemEval-inspired project by
combining bias detection, retrieval-based explanations, and
interpretable machine learning.