import pandas as pd, numpy as np, re, json, warnings, time
warnings.filterwarnings("ignore")
import os; exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"week3_validation_experiments.py")).read().split("R=[]")[0])   # reuse data prep, vec(), helpers
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import StandardScaler
LR=lambda C=1,**k:LogisticRegression(C=C,max_iter=2000,solver="liblinear",random_state=42,**k)
models={
 "Majority class":Pipeline([("v",vec()),("c",DummyClassifier(strategy="most_frequent"))]),
 "LR, unigram TF-IDF":Pipeline([("v",vec()),("c",LR())]),
 "LR, uni+bigram, C=3":Pipeline([("v",vec(ngram_range=(1,2))),("c",LR(3))]),
 "Linear SVM, uni+bigram, C=0.3":Pipeline([("v",vec(ngram_range=(1,2))),("c",LinearSVC(C=0.3,random_state=42))]),
 "Complement NB, uni+bigram, a=1":Pipeline([("v",vec(ngram_range=(1,2))),("c",ComplementNB(alpha=1.0))]),
 "LR, uni+bigram, chi2 top-5000":Pipeline([("v",vec(ngram_range=(1,2))),("s",SelectKBest(chi2,k=5000)),("c",LR(1))]),
 "MLP (128 hidden), uni+bigram":Pipeline([("v",vec(ngram_range=(1,2))),("c",MLPClassifier((128,),alpha=1e-2,max_iter=30,early_stopping=True,random_state=42))]),
}
rng=np.random.default_rng(42); idx=[rng.integers(0,len(yte),len(yte)) for _ in range(2000)]
out={}; P={}
for k,m in models.items():
    m.fit(tr.text,ytr); p=m.predict(te.text); P[k]=p
    f=f1_score(yte,p,average="macro"); bs=[f1_score(yte[i],p[i],average="macro") for i in idx]
    out[k]=dict(macro_f1=f,ci=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))],acc=accuracy_score(yte,p))
    print(f"{k:35s} test F1={f:.4f} [{out[k]['ci'][0]:.3f},{out[k]['ci'][1]:.3f}] acc={out[k]['acc']:.4f}")
# paired bootstrap vs LR default
def pdiff(a,b):
    d=[f1_score(yte[i],P[a][i],average="macro")-f1_score(yte[i],P[b][i],average="macro") for i in idx]; return float(np.mean(d)),[float(np.percentile(d,2.5)),float(np.percentile(d,97.5))]
out["paired"]={k:pdiff(k,"LR, unigram TF-IDF") for k in ["LR, uni+bigram, C=3","Linear SVM, uni+bigram, C=0.3","Complement NB, uni+bigram, a=1","LR, uni+bigram, chi2 top-5000","MLP (128 hidden), uni+bigram"]}
print(out["paired"])
# MLP seeds
ms=[]
for s in [1,2,3,4,5]:
    m=Pipeline([("v",vec(ngram_range=(1,2))),("c",MLPClassifier((128,),alpha=1e-2,max_iter=30,early_stopping=True,random_state=s))]).fit(tr.text,ytr)
    ms.append(f1_score(yva,m.predict(va.text),average="macro"))
out["mlp_val_seeds"]=[round(x,4) for x in ms]; print("MLP val over 5 seeds",out["mlp_val_seeds"])
# --- feature engineering: style features + text
def style(d):
    s=d.text
    return np.c_[s.str.len(),s.str.split().str.len(),s.str.count(r"\d"),s.str.count("!"),s.str.count(r"[A-Z]")/s.str.len().clip(lower=1),s.str.count(r"\b(?:not|no|never)\b"),s.str.contains(r"\d+\s*percent|%").astype(int)]
sc2=StandardScaler().fit(style(tr)); 
v=vec(ngram_range=(1,2)); Xt=v.fit_transform(tr.text); Xv=v.transform(va.text); Xe=v.transform(te.text)
H=lambda X,d:hstack([X,csr_matrix(sc2.transform(style(d)))]).tocsr()
m=LR(3).fit(H(Xt,tr),ytr); out["style_plus_text_val"]=f1_score(yva,m.predict(H(Xv,va)),average="macro"); out["style_only_val"]=f1_score(yva,LR().fit(sc2.transform(style(tr)),ytr).predict(sc2.transform(style(va))),average="macro")
print("text+style val",out["style_plus_text_val"],"style only",out["style_only_val"])
# --- leakage: tree on credit counts (own-label proportion)
cc=["c_bt","c_f","c_ht","c_mt","c_pf"]
def cf(d):
    c=d[cc].astype(float).values; tot=c.sum(1,keepdims=True); return np.c_[c,tot,c/np.maximum(tot,1),(tot==0)]
g=HistGradientBoostingClassifier(random_state=42).fit(cf(tr),ytr)
out["credit_tree_val"]=f1_score(yva,g.predict(cf(va)),average="macro"); out["credit_tree_test"]=f1_score(yte,g.predict(cf(te)),average="macro")
print("credit-history tree val/test",out["credit_tree_val"],out["credit_tree_test"])
# --- six-way
y6=lambda d:d.label.values
m6=Pipeline([("v",vec(ngram_range=(1,2))),("c",LogisticRegression(C=3,max_iter=1000))]).fit(tr.text,y6(tr)); p6=m6.predict(te.text)
out["sixway_test_macro_f1"]=f1_score(y6(te),p6,average="macro"); out["sixway_test_acc"]=accuracy_score(y6(te),p6); out["sixway_majority_acc"]=float((y6(te)==pd.Series(y6(tr)).mode()[0]).mean())
print("six-way",out["sixway_test_macro_f1"],out["sixway_test_acc"],out["sixway_majority_acc"])
# --- learning curve for LR C=3
lc=[]
for frac in [0.1,0.25,0.5,0.75,1.0]:
    sub=tr.sample(frac=frac,random_state=42) if frac<1 else tr
    m=Pipeline([("v",vec(ngram_range=(1,2))),("c",LR(3))]).fit(sub.text,y(sub)); lc.append((frac,len(sub),f1_score(yva,m.predict(va.text),average="macro")))
out["learning_curve"]=lc; print(lc)
# --- error analysis with best val model: LR C=3 (chosen); confusion + top features
best=models["Linear SVM, uni+bigram, C=0.3"]; pb=P["Linear SVM, uni+bigram, C=0.3"]
from sklearn.metrics import confusion_matrix
out["confusion_svm"]=confusion_matrix(yte,pb).tolist()
lr=models["LR, uni+bigram, C=3"]; names=lr.named_steps["v"].get_feature_names_out(); co=lr.named_steps["c"].coef_[0]
o=np.argsort(co); out["top_fake_feats"]=[names[i] for i in o[::-1][:15]]; out["top_real_feats"]=[names[i] for i in o[:15]]
print("fake feats",out["top_fake_feats"]);print("real feats",out["top_real_feats"])
# six-way error: where mistakes fall by original label
te2=te.copy(); te2["pred"]=pb; te2["ok"]=(pb==yte)
out["acc_by_label"]=te2.groupby("label").ok.mean().round(3).to_dict(); print(out["acc_by_label"])
dec=best.decision_function(te.text); te2["conf"]=np.abs(dec)
err=te2[~te2.ok].sort_values("conf",ascending=False)
sel=pd.concat([err[err.bin=="real"].head(8),err[err.bin=="fake"].head(8)])
sel[["label","pred","conf","statement","speaker","subject"]].to_csv("docs/week3/error_samples.csv",index=False)
for _,r in sel.iterrows(): print(f"[{r.label:11s} pred={'fake' if r.pred else 'real'} c={r.conf:.2f}] {r.speaker[:18]:18s} | {r.statement[:150]}")
json.dump(out,open("docs/week3/test_results.json","w"),indent=1,default=str)
