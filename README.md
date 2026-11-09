# TruthLens: English Fake News Detection

A cross-dataset fake news detector for English text, built as part of the Junior Natural Language Processing Specialist internship at YuvaIntern.

> **Status: Week 1 (planning) complete.** This repository currently holds the strategic plan and literature review. Code will be added in Weeks 2 to 4.

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
├── docs/            # Week 1 report (plan and literature review)
├── data/            # datasets (not committed)
├── notebooks/       # experiments
└── src/             # pipeline and model code
```

Raw datasets and trained model weights are not committed because of their size. Download links and instructions will be added here once the data pipeline is built in Week 2.

## Tools

Python, pandas, scikit-learn, Hugging Face Transformers, PyTorch, LIME, Streamlit, Git and GitHub, Google Colab or Kaggle for GPU time.

## Author

Anshuman Tripathy, B.Tech CSE, Amity University Kolkata.
