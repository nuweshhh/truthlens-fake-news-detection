"""Charts for the Week 4 report. Run after week4_evaluation.py and week4_lime.py."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

O = json.load(open("docs/week4/evaluation_results.json"))
C = json.load(open("docs/week4/curves.json"))
L = json.load(open("docs/week4/lime_explanations.json"))
os.makedirs("docs/figures", exist_ok=True)
B, R, G = "#2a6f97", "#c4553a", "#8a8f98"
cols = {"LR bigram C=3": "#2a6f97", "Linear SVM C=0.3": "#c4553a", "Complement NB a=1": "#6b8e23", "MLP 128": "#7b5ea7"}

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for k, c in C.items():
    ax[0].plot(c["fpr"], c["tpr"], color=cols[k], label=f"{k} (AUC {O[k]['auc']:.3f})", lw=1.5)
    ax[1].plot(c["rec"], c["prec"], color=cols[k], lw=1.5, label=f"{k} (AP {O[k]['ap']:.3f})")
ax[0].plot([0, 1], [0, 1], ls="--", color=G, lw=1)
ax[0].set_xlabel("False-positive rate"); ax[0].set_ylabel("True-positive rate (fake recall)")
ax[0].set_title("ROC curves, test set", fontsize=10); ax[0].legend(fontsize=7, loc="lower right")
ax[1].axhline(O["ap_baseline"], ls="--", color=G, lw=1)
ax[1].text(.02, O["ap_baseline"] + .01, "no-skill = fake share 0.433", fontsize=7, color="#555")
ax[1].set_xlabel("Recall (fake)"); ax[1].set_ylabel("Precision (fake)")
ax[1].set_title("Precision-recall curves, test set", fontsize=10); ax[1].legend(fontsize=7, loc="upper right"); ax[1].set_ylim(.3, 1)
plt.tight_layout(); plt.savefig("docs/figures/w4_roc_pr.png", dpi=160); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.6))
for a, k in zip(ax, ["Linear SVM C=0.3", "MLP 128"]):
    cm = np.array(O[k]["cm"]); n = cm / cm.sum(1, keepdims=True)
    a.imshow(n, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            a.text(j, i, f"{cm[i, j]}\n({n[i, j] * 100:.0f}%)", ha="center", va="center",
                   color="white" if n[i, j] > .55 else "#222", fontsize=10)
    a.set_xticks([0, 1]); a.set_xticklabels(["pred real", "pred fake"])
    a.set_yticks([0, 1]); a.set_yticklabels(["true real", "true fake"]); a.set_title(k + " (row %)", fontsize=10)
plt.tight_layout(); plt.savefig("docs/figures/w4_confusion.png", dpi=160); plt.close()

want = ["<=10 words", "11-17 words", "18-25 words", ">25 words", "no digit", "contains a digit", "contains percent/%",
        "contains negation (not/no/never/n't)", "starts with 'Says'", "mentions Obama/Clinton/Trump/Romney/Bush",
        "party: republican", "party: democrat"]
S = {s["group"]: s for s in O["slices"]}
lab = {"contains negation (not/no/never/n't)": "has not/no/never",
       "mentions Obama/Clinton/Trump/Romney/Bush": "names Obama/Clinton/Trump/Romney/Bush",
       "contains percent/%": "has percent or %", "starts with 'Says'": "starts with 'Says'"}
fig, ax = plt.subplots(figsize=(8.6, 5.2)); y = np.arange(len(want))[::-1]
for yy, w in zip(y, want):
    s = S[w]; ax.plot([s["fpr"], s["fake_recall"]], [yy, yy], color="#c9ccd1", lw=2, zorder=1)
    ax.scatter(s["fake_recall"], yy, color=B, zorder=2, s=40); ax.scatter(s["fpr"], yy, color=R, zorder=2, s=40)
ax.set_yticks(y); ax.set_yticklabels([f"{lab.get(w, w)}  (n={S[w]['n']})" for w in want], fontsize=8)
ax.set_xlim(0, .7); ax.set_xlabel("Rate")
ax.scatter([], [], color=B, label="fake recall (fake statements caught)")
ax.scatter([], [], color=R, label="false-positive rate (real flagged as fake)")
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.12), ncol=2, frameon=False)
ax.set_title("Digits and length make the model call statements real", fontsize=10); ax.grid(axis="x", alpha=.3)
plt.tight_layout(); plt.savefig("docs/figures/w4_slices.png", dpi=160); plt.close()

fig, axs = plt.subplots(2, 2, figsize=(10, 5.4))
for a, e in zip(axs.ravel(), L):
    f = e["feats"][::-1]
    a.barh([w for w, _ in f], [v for _, v in f], color=[R if v > 0 else B for _, v in f])
    a.axvline(0, color="#444", lw=.8)
    a.set_title(f"{e['title']} ({e['label']}, P(fake)={e['pred_fake_prob']})", fontsize=8.5); a.tick_params(labelsize=8)
fig.text(.5, .005, "Red = pushes towards fake, blue = pushes towards real (LIME weight, logistic regression)", ha="center", fontsize=8)
plt.tight_layout(rect=(0, .03, 1, 1)); plt.savefig("docs/figures/w4_lime.png", dpi=160); plt.close()
