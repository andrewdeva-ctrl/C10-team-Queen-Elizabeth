# C10-team-Queen Elizabeth
# AgriSLM TRI Project — Team QueenElizabeth  
Sri Lanka + South Africa Agricultural Small Language Model

## Overview
This repository contains our submission for TRI AI Cohort 10.  
We fine‑tune a Small Language Model (SLM) to generate concise, extension‑style answers for farmers facing crop, soil, livestock, and climate challenges.

Our work combines the official Kaggle Agriculture & Climate SLM dataset with a small synthetic dataset we previously created for our Sri Lanka + South Africa AgriSLM project.

---

## Dataset Overview

### 1. Kaggle Agriculture & Climate SLM Benchmark
We use the official Kaggle dataset, which includes:
- 24 synthetic extension documents  
- training Q&A pairs  
- test questions  
- Levenshtein‑distance scoring  

Files:

documents.csv
train_qa.csv
test_questions.csv
sample_submission.csv
dataset-metadata.json
submission.csv

---

## 2. Team-Generated Parallel Dataset (Sri Lanka + South Africa)

### Why This Dataset Exists
Before TRI, our team was already working on a **Sri Lanka + South Africa Agricultural SLM (AgriSLM)** project.  
During that work, we created several **synthetic Q&A datasets** covering crop diseases, soil issues, climate adaptation, and livestock management.

To support TRI’s educational goals, we **aligned our existing synthetic datasets to the Kaggle benchmark**.  
This ensures:
- compatibility with Kaggle’s Q&A structure  
- consistency with TRI’s evaluation format  
- clear separation between Kaggle data and team-generated data  
- transparent documentation of our original project work  

### Included Files
data/agrislm_team_demo_dataset.csv
data/agrislm_team_demo_dataset.jsonl
parallel_data.md

Our dataset is fully synthetic, manually authored, and designed to complement Kaggle by adding localized Sri Lanka + South Africa scenarios.

## Repository Structure

C10–team–queenelizabeth/
├── README.md
├── docs/
│   ├── problem_statement.pdf
│   ├── data_card.pdf
│   ├── impact_statement_card.pdf
│   └── stakeholder_engagement.pdf
├── scripts/
│   └── kaggle_pipeline.py
└── data/
├── documents.csv
├── train_qa.csv
├── test_questions.csv
├── sample_submission.csv
├── dataset-metadata.json
├── agrislm_team_demo_dataset.csv
└── parallel_data.md

---

## How to Run
Install dependencies:
pip install -r requirements.txt

Run baseline pipeline:
python scripts/kaggle_pipeline.py

This generates a submission CSV for the Kaggle test set.

---

## Team Members
- Ernest Nchabeng (Leader)  
- Andrew Devadason  

## License
Team-generated synthetic data: **CC BY-SA 4.0**  
Kaggle dataset: original license.

