# Data

Raw data is not committed (see `.gitignore`). Download it into `data/raw/`.

## LIAR (used in Week 2)

Train, validation and test files from the public LIAR-PLUS release:

```
mkdir -p data/raw/liar
B=https://raw.githubusercontent.com/Tariq60/LIAR-PLUS/master/dataset/tsv
for f in train2 val2 test2; do curl -L -o data/raw/liar/$f.tsv $B/$f.tsv; done
```

Dataset: LIAR-PLUS v2 (Tariq60/LIAR-PLUS; Alhindi, Petridis & Muresan, 2018, FEVER workshop). Each row has 16 tab-separated fields: the leading row index, the 14 original LIAR fields (id, label, statement, subject, speaker, job title, state, party, five credit-history counts, context) and the added `justification` field. Only the statement text is used as model input; `justification` is excluded because it leaks the verdict.

Parsing notes:

- **Empty continuation rows.** 47 rows (31 in train, 16 in test) are empty continuation rows. A statement whose text contains a line break is written over two lines; the first line holds the id, label and statement, and the next line is only a tab (`\t`). Read as a table, that second line becomes a row with every field empty. These records were not reconstructed: the statement and label stay in the preceding row, and the EDA script drops the rows with an empty label. All 12,836 labelled statements are kept (10,269 / 1,284 / 1,283).
- **Quotation marks.** The files are read with `quoting=3` (no CSV quote handling), so quotation marks stay in the text: 523 statements begin with a quotation mark and 521 contain doubled quotation marks (`""`). The text is not normalised for this.

## ISOT

Download "Fake and Real News Dataset" (ISOT, University of Victoria; also on Kaggle) and place `True.csv` and `Fake.csv` in `data/raw/isot/`. Check the dataset licence before use.
