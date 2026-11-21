# TruthLens: English Fake News Detection

A cross-dataset fake news detector for English text, built as part of the Junior Natural Language Processing Specialist internship at YuvaIntern.

> **Status: Week 4 (evaluation, error analysis and final report) complete.** The repository holds the Week 1 plan, the cleaning code, an exploratory analysis of LIAR, classical baseline experiments, and the final evaluation with error analysis. DistilBERT fine-tuning, the ISOT cross-dataset test and an interactive demo were not completed (see Known limitations).

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
- **LIAR**: short fact-checked statements from PolitiFact, mapped from six truth levels to binary (real vs fake). The copy used is LIAR-PLUS v2 (Tariq60/LIAR-PLUS) with 16 TSV fields: the row index, the 14 original LIAR fields and the added justification field. Only the statement text is the model input; justification is excluded (Alhindi, Petridis & Muresan, 2018, FEVER workshop).
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

- Quotation marks are kept as written in the TSV files: 523 statements begin with one and 521 contain doubled quotation marks (`""`); no normalisation is applied.
- 47 rows (31 train, 16 test) are empty continuation rows from statements split over two lines; the statement and label stay in the preceding row and the empty rows are dropped.
- Single random seed, so differences between runs are indicative, not statistically tested.
- ISOT (articles) and LIAR (short claims) differ in format, so the cross-dataset test also measures format shift.
- No fresh test set is built, so performance on future news is not measured.
- ISOT, DistilBERT fine-tuning and the interactive demo were not completed in the available environment (no ISOT download, no model downloads, no GPU), so results are for LIAR and classical models only.

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

## Week 3 results (LIAR, text only)

- Models compared: majority class, logistic regression, linear SVM, Multinomial and Complement Naive Bayes, a small MLP; 27 configurations scored on the validation set.
- Test macro-F1 (1,279 statements, 95% bootstrap intervals): majority 0.362; unigram logistic regression 0.597; linear SVM 0.604; MLP 0.607. All trained models are statistically indistinguishable (every paired difference interval contains zero).
- Removing stop words was the only preprocessing change that clearly hurt (-0.025 validation macro-F1).
- A tree model on the speaker credit-history columns alone reaches 0.716 test macro-F1, above every text model, because those counts include the current statement. Text-only inputs are therefore used.
- The DistilBERT fine-tuning and ISOT experiments have not been run yet.
- Reproduce: `python src/eda_liar.py`, `python src/week3_validation_experiments.py`, `python src/week3_test_evaluation.py`, `python src/week3_figures.py` (outputs in `docs/week3/` and `docs/figures/`).

## Week 4 results (LIAR, text only)

- Final test metrics (1,279 statements): the five trained models reach 0.597 to 0.607 macro-F1, 62% to 63% accuracy and ROC-AUC 0.66 to 0.67. They find 44% to 46% of fake statements and 75% to 77% of real ones.
- 363 test statements (28%) are misclassified by all four of the best models, so most errors are not caused by the choice of algorithm.
- Fake recall is 0.32 for statements containing a digit and 0.52 for those without; it also falls with statement length.
- Tested improvements: threshold tuning on validation gives +0.004 to +0.016 macro-F1; masking "Says", numbers or names makes no detectable difference; five-fold cross-validation shows the single validation split overstated two models by 0.024 to 0.027.
- Reproduce: after the Week 3 scripts, run `python src/week4_evaluation.py`, `python src/week4_cue_masking.py`, `python src/week4_lime.py` and `python src/week4_figures.py` (outputs in `docs/week4/` and `docs/figures/`). `lime` is required for the explanations.

## Tools

Python, pandas, scikit-learn, Hugging Face Transformers, PyTorch, LIME, Streamlit, Git and GitHub, Google Colab or Kaggle for GPU time.

## Author

Anshuman Tripathy, B.Tech CSE, Amity University Kolkata.
