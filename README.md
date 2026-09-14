# AgriSLM — Team QueenElizabeth

**TRI AI Cohort 10 | Sri Lanka + South Africa Agricultural Small Language Model**

AgriSLM is a responsible-AI research project exploring how a compact, agriculture-focused language model can improve access to clear, localized and climate-aware agricultural guidance for smallholder farmers in Sri Lanka and South Africa.

> **Research prototype:** AgriSLM is not a substitute for qualified agronomic, veterinary, pesticide-label, extension or official weather guidance.

## Dataset

The project uses two clearly separated data sources.

**1. TRI/Kaggle Agriculture & Climate SLM benchmark**
- `documents.csv` — 24 synthetic agricultural extension documents.
- `train_qa.csv` — 45 training question-answer pairs.
- `test_questions.csv` — 12 unseen test questions.
- `baseline_submission.csv` and `dataset-metadata.json` — benchmark support files.

**2. Team-generated AgriSLM demo data**
- `data/agrislm_team_demo_dataset.csv`
- `data/agrislm_team_demo_dataset.jsonl`
- `data/parallel_data.md`

The team-generated records are synthetic and manually authored to represent Sri Lanka + South Africa agricultural scenarios. They are demo/research material, not validated field prescriptions. Benchmark data and team-generated data remain separate in provenance and licensing.

## Training Pipeline

The current reproducible baseline uses **TF-IDF retrieval** rather than claiming a fully fine-tuned production SLM.

1. Load competition documents, training Q&A and test questions.
2. Preprocess text using lowercasing, tokenization and stopword removal.
3. Build TF-IDF representations of the extension content.
4. Compute cosine similarity between each test question and candidate source content.
5. Retrieve the most relevant content and produce concise answers in `QuestionId,Answer` format.
6. Save the final submission CSV for Kaggle evaluation.

**Model/design choice:** TF-IDF was selected as a transparent, lightweight baseline that is easy to reproduce and compare. No exhaustive hyperparameter search was performed; the baseline prioritised reproducibility and retrieval quality. Future work will compare stronger retrieval/RAG and compact-model adaptation approaches.

Notebook: [`scripts/queenelizabeth-tfidf-submission.ipynb`](scripts/queenelizabeth-tfidf-submission.ipynb)

Kaggle notebook: https://www.kaggle.com/code/andrewdevadason/queenelizabeth-tfidf-submission

## Evaluation

The method is evaluated using the TRI benchmark metric: **mean character-level Levenshtein distance** between generated answers and hidden reference answers. Lower scores are better.

Evaluation process:
1. Generate answers for all 12 test questions.
2. Validate the output schema as exactly `QuestionId,Answer`.
3. Check that all test IDs are present in order and answers are non-empty.
4. Submit the CSV to Kaggle for scoring against the hidden test set.

## Reproduction

1. Join the Agriculture & Climate SLM challenge on Kaggle.
2. Open the notebook in `scripts/` or the Kaggle notebook linked above.
3. Attach/load the competition files: `documents.csv`, `train_qa.csv`, `test_questions.csv`, `baseline_submission.csv` and `dataset-metadata.json`.
4. Run all notebook cells from top to bottom.
5. Confirm that `/kaggle/working/submission.csv` is created.
6. Verify the CSV contains 12 rows plus the header and uses `QuestionId,Answer`.
7. Submit the generated file through the Kaggle competition workflow.

## Cohort Challenge Documentation

The `docs/` folder contains all four required Cohort Challenges:

- `problem_statement.pdf` — Problem Statement / research gaps and values-led framework.
- `data_card.pdf` — Ethical Dataset Data Card.
- `impact_statement_card.pdf` — Impact Statement Card.
- `stakeholder_engagement.pdf` — Stakeholder Engagement Plan.

These documents cover the project problem, ethical data governance, anticipated impacts/risks and stakeholder participation.

## Repository Structure

```text
C10-team-Queen-Elizabeth/
├── README.md
├── data/
│   ├── documents.csv
│   ├── train_qa.csv
│   ├── test_questions.csv
│   ├── baseline_submission.csv
│   ├── dataset-metadata.json
│   ├── agrislm_team_demo_dataset.csv
│   ├── agrislm_team_demo_dataset.jsonl
│   ├── parallel_data.md
│   └── submission.csv
├── docs/
│   ├── problem_statement.pdf
│   ├── data_card.pdf
│   ├── impact_statement_card.pdf
│   └── stakeholder_engagement.pdf
└── scripts/
    └── queenelizabeth-tfidf-submission.ipynb
```

## Responsible Use

AgriSLM should interpret trusted climate/weather information rather than claim to independently forecast weather. High-risk pesticide, fertiliser, disease or livestock guidance should use authoritative, locally applicable sources and qualified review where needed. Traditional/community knowledge should retain consent, provenance and evidence-status information.

## Appendix — Contributors and Mentors

**Team QueenElizabeth**
- Ernest Nchabeng — Team Leader
- Andrew Devadason — Team Member

**Mentor(s):** Add the assigned TRI mentor name(s) before final submission if required.

## License

Team-generated synthetic AgriSLM data: **CC BY-SA 4.0**. TRI/Kaggle benchmark files retain their original source terms and licensing.
