"""Charts for the Week 3 report. Run after week3_validation_experiments.py and week3_test_evaluation.py."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T = json.load(open("docs/week3/test_results.json"))
V = pd.DataFrame(json.load(open("docs/week3/validation_results.json")))
S = json.load(open("docs/week3/shortcut_results.json"))
G, B, R = "#8a8f98", "#2a6f97", "#c4553a"
import os
os.makedirs("docs/figures", exist_ok=True)

names = ["Majority class", "LR, unigram TF-IDF", "LR, uni+bigram, C=3", "Linear SVM, uni+bigram, C=0.3",
         "Complement NB, uni+bigram, a=1", "LR, uni+bigram, chi2 top-5000", "MLP (128 hidden), uni+bigram"]
f = [T[n]["macro_f1"] for n in names]
lo = [T[n]["macro_f1"] - T[n]["ci"][0] for n in names]
hi = [T[n]["ci"][1] - T[n]["macro_f1"] for n in names]
fig, ax = plt.subplots(figsize=(8.5, 3.9)); y = np.arange(len(names))[::-1]
ax.errorbar(f, y, xerr=[lo, hi], fmt="o", color=B, ecolor=G, capsize=3)
ax.errorbar([f[0]], [y[0]], xerr=[[lo[0]], [hi[0]]], fmt="o", color=G, ecolor=G, capsize=3)
for yy, ff in zip(y, f):
    ax.text(ff, yy + .22, f"{ff:.3f}", ha="center", fontsize=8)
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel("Test macro-F1 (95% bootstrap interval)"); ax.set_xlim(.33, .67); ax.grid(axis="x", alpha=.3)
ax.set_title("All trained models sit in one band, far above majority class", fontsize=10)
plt.tight_layout(); plt.savefig("docs/figures/fig5_models.png", dpi=160); plt.close()

base = V[V.name == "LR, unigram TF-IDF (default)"].macro_f1.iloc[0]
ab = ["LR, uni+bigram", "LR, unigram, Porter stemming", "LR, char 2-5 grams", "LR, unigram, no lowercasing",
      "LR, unigram, class_weight=balanced", "LR, unigram, stop words removed but negations kept",
      "LR, unigram, sklearn stop words removed"]
lab = ["add bigrams", "Porter stemming", "char 2-5 grams", "keep case (no lowercasing)", "class weights balanced",
       "stop words removed, negations kept", "stop words removed (sklearn list)"]
d = [V[V.name == n].macro_f1.iloc[0] - base for n in ab]
fig, ax = plt.subplots(figsize=(8.5, 3.6)); y = np.arange(len(ab))[::-1]
ax.barh(y, d, color=[R if x < 0 else B for x in d]); ax.axvline(0, color="#444", lw=.8)
for yy, x in zip(y, d):
    ax.text(x + (.0008 if x >= 0 else -.0008), yy, f"{x:+.3f}", va="center", ha="left" if x >= 0 else "right", fontsize=8)
ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=8)
ax.set_xlabel(f"Change in validation macro-F1 vs unigram LR default ({base:.3f})"); ax.set_xlim(-.04, .03)
ax.set_title("Removing stop words is the only change that clearly hurts", fontsize=10)
plt.tight_layout(); plt.savefig("docs/figures/fig6_ablation.png", dpi=160); plt.close()

labs = ["Text only (SVM, test)", "Speaker + party + subject only (val)", "Credit-history counts, tree (test)", "Majority class (test)"]
v = [T["Linear SVM, uni+bigram, C=0.3"]["macro_f1"], S["speaker+party+subject only"]["macro_f1"],
     T["credit_tree_test"], T["Majority class"]["macro_f1"]]
fig, ax = plt.subplots(figsize=(8.5, 3.2)); y = np.arange(4)[::-1]
ax.barh(y, v, color=[B, G, R, "#c9ccd1"])
for yy, x in zip(y, v):
    ax.text(x + .005, yy, f"{x:.3f}", va="center", fontsize=9)
ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=9); ax.set_xlim(0, .85); ax.set_xlabel("Macro-F1")
ax.set_title("Credit-history columns beat the text model: they leak the label", fontsize=10)
plt.tight_layout(); plt.savefig("docs/figures/fig7_leakage.png", dpi=160); plt.close()
