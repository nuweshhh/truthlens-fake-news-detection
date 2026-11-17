# Data

Raw data is not committed (see `.gitignore`). Download it into `data/raw/`.

## LIAR (used in Week 2)

Train, validation and test files from the public LIAR-PLUS release:

```
mkdir -p data/raw/liar
B=https://raw.githubusercontent.com/Tariq60/LIAR-PLUS/master/dataset/tsv
for f in train2 val2 test2; do curl -L -o data/raw/liar/$f.tsv $B/$f.tsv; done
```

The files contain a few blank lines from line breaks inside the justification column. The EDA script drops rows with an empty label, which gives 10,269 / 1,284 / 1,283 statements.

## ISOT

Download "Fake and Real News Dataset" (ISOT, University of Victoria; also on Kaggle) and place `True.csv` and `Fake.csv` in `data/raw/isot/`. Check the dataset licence before use.
