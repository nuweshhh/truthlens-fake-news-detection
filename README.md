# TruthLens: English Fake News Detection

A cross-dataset fake news detector for English text, built as part of the Junior Natural Language Processing Specialist internship at YuvaIntern.

> **Status: Week 2 (data preprocessing and exploratory analysis) complete.** The repository holds the Week 1 plan, the cleaning code and an exploratory analysis of LIAR. Modelling follows in Weeks 3 and 4.

## Problem

False news spreads faster than true news, so manual fact-checking cannot keep up at scale. This project builds a transformer-based classifier that labels English news text as real or fake, tests whether it still works on data it has never seen, and shows which words influenced each prediction.

## Objectives

| # | Objective | Success test |
| --- | --- | --- |
| O1 | Reproducible data pipeline for two public datasets | One script rebuilds all train, validation and test splits |
| O2 | Baselines established | TF-IDF with logistic regression and SVM scores recorded on every dataset |
| O3 | Transformer beats the baselines | Outperforms the best classical baseline on in-domain macro-F1 |
| O4 | Generalisation measured | Cross-dataset macro-F1 reported, with the drop quantified |
| O5 | Errors understood | 10 to 15 representative misclassified samples labelled by cause |
| O6 | Explainable demo | Web demo returns label, confidence and highlighted words |

## Datasets

- **ISOT**: full news articles (real vs fake).
- **LIAR**: short fact-checked statements from PolitiFact, mapped from six truth levels to binary (real vs fake).
- **FakeNewsNet** (optional): only through a pre-cleaned public copy.

## Planned approach

1. **Data collection:** download ISOT and LIAR and record their licences.
2. **Preprocessing:** map labels, remove duplicates across splits, strip source-specific shortcuts, use fixed seeds and stratified splits.
3. **Modelling:** TF-IDF baselines, then a fine-tuned DistilBERT model (single seed, 42). Optional stretch goals: BERT-base and an LLM prompting baseline.
4. **Evaluation:** in-domain test, cross-dataset test, error analysis and LIME word highlights.
5. **Reporting:** results report, Streamlit demo and a short model card.

## Roadmap

| Week | Focus | Exit gate |
| --- | --- | --- |
| 1 | Plan and literature review | Plan delivered, datasets confirmed available |
| 2 | Data pipeline and baselines | One script rebuilds splits, baseline scores on every dataset |
| 3 | DistilBERT and cross-dataset test | Transformer beats best baseline, cross-dataset scores reported |
| 4 | Explainability, demo and final report | Errors labelled, demo running, report complete |

## Evaluation metrics

Macro-F1 is the headline metric. Accuracy, precision, recall and ROC-AUC are also reported, together with the cross-dataset drop and a simple speed check.

## Known limitations

- Single random seed, so differences between runs are indicative, not statistically tested.
- ISOT (articles) and LIAR (short claims) differ in format, so the cross-dataset test also measures format shift.
- No fresh test set is built, so performance on future news is not measured.

## Repository structure

```
truthlens-fake-news-detection/
├── .gitignore       # keeps datasets, model weights and environments out of Git
├── README.md
├── docs/            # reports and EDA figures
├── data/            # datasets (not committed)
├── notebooks/       # experiments
└── src/             # pipeline and model code
```

Raw datasets and trained model weights are not committed because of their size. Download instructions are in `data/README.md`.

## Week 2 results (LIAR)

- 12,836 statements (10,269 / 1,284 / 1,283); binary mapping gives 7,167 real and 5,669 fake.
- Job title and state are empty in 27.9% and 21.4% of rows; text is already clean (no URLs or HTML), so cleaning changes only 34 statements.
- Length differs only slightly between classes (mean 18.7 vs 17.6 words, effect size 0.078).
- Credit-history columns include the current statement (99.7% of applicable rows), so they are excluded from text models.
- Run `python src/eda_liar.py` after downloading the data to reproduce the figures in `docs/figures/` and the summary in `docs/week2_liar_eda_summary.json`.

## Tools

Python, pandas, scikit-learn, Hugging Face Transformers, PyTorch, LIME, Streamlit, Git and GitHub, Google Colab or Kaggle for GPU time.

## Author

Anshuman Tripathy, B.Tech CSE, Amity University Kolkata.
