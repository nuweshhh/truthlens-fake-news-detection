import pandas as pd, numpy as np, re, json, warnings
warnings.filterwarnings("ignore")
import os; exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"week3_validation_experiments.py")).read().split("R=[]")[0])
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import (precision_recall_fscore_support, roc_auc_score, average_precision_score, roc_curve,
                             precision_recall_curve, confusion_matrix)
from sklearn.model_selection import StratifiedKFold, cross_val_score
LR=lambda C=1,**k:LogisticRegression(C=C,max_iter=2000,solver="liblinear",random_state=42,**k)
V2=lambda:vec(ngram_range=(1,2))
M={"Majority class":Pipeline([("v",vec()),("c",DummyClassifier(strategy="most_frequent"))]),
 "LR unigram":Pipeline([("v",vec()),("c",LR())]),
 "LR bigram C=3":Pipeline([("v",V2()),("c",LR(3))]),
 "Linear SVM C=0.3":Pipeline([("v",V2()),("c",LinearSVC(C=0.3,random_state=42))]),
 "Complement NB a=1":Pipeline([("v",V2()),("c",ComplementNB(alpha=1.0))]),
 "MLP 128":Pipeline([("v",V2()),("c",MLPClassifier((128,),alpha=1e-2,max_iter=30,early_stopping=True,random_state=42))])}
def score(m,X):
    c=m.named_steps["c"]
    return c.decision_function(m.named_steps["v"].transform(X)) if hasattr(c,"decision_function") else c.predict_proba(m.named_steps["v"].transform(X))[:,1]
rng=np.random.default_rng(42); idx=[rng.integers(0,len(yte),len(yte)) for _ in range(2000)]
O={"n_test":len(yte),"n_fake":int(yte.sum())}; P={};SC={}
for k,m in M.items():
    m.fit(tr.text,ytr); p=m.predict(te.text); s=score(m,te.text); P[k]=p; SC[k]=s
    pr,rc,f,_=precision_recall_fscore_support(yte,p,labels=[0,1],zero_division=0)
    O[k]=dict(acc=accuracy_score(yte,p),macro_f1=f1_score(yte,p,average="macro"),weighted_f1=f1_score(yte,p,average="weighted"),
              real_p=pr[0],real_r=rc[0],real_f1=f[0],fake_p=pr[1],fake_r=rc[1],fake_f1=f[1],
              auc=roc_auc_score(yte,s),ap=average_precision_score(yte,s),cm=confusion_matrix(yte,p).tolist())
    print(f"{k:20s} acc={O[k]['acc']:.3f} mF1={O[k]['macro_f1']:.3f} fakeP={pr[1]:.3f} fakeR={rc[1]:.3f} realP={pr[0]:.3f} realR={rc[0]:.3f} AUC={O[k]['auc']:.3f} AP={O[k]['ap']:.3f}")
O["ap_baseline"]=float(yte.mean())
# bootstrap CI for AUC of svm
O["auc_ci_svm"]=[float(np.percentile([roc_auc_score(yte[i],SC["Linear SVM C=0.3"][i]) for i in idx[:1000]],q)) for q in (2.5,97.5)]
# ---- curves data
curves={}
for k in ["LR bigram C=3","Linear SVM C=0.3","Complement NB a=1","MLP 128"]:
    fpr,tpr,_=roc_curve(yte,SC[k]); pr_,rc_,_=precision_recall_curve(yte,SC[k]); curves[k]=dict(fpr=fpr.tolist(),tpr=tpr.tolist(),prec=pr_.tolist(),rec=rc_.tolist())
json.dump(curves,open("docs/week4/curves.json","w"))
# ---- error overlap
names=["LR bigram C=3","Linear SVM C=0.3","Complement NB a=1","MLP 128"]
wrong=np.array([P[k]!=yte for k in names]); 
O["wrong_all4"]=int(wrong.all(0).sum()); O["wrong_any"]=int(wrong.any(0).sum()); O["wrong_none"]=int((~wrong.any(0)).sum()); O["wrong_each"]=[int(w.sum()) for w in wrong]
vote=(np.sum([P[k] for k in names],axis=0)>=2).astype(int); O["vote_macro_f1"]=f1_score(yte,vote,average="macro")
print("wrong all4/any/none",O["wrong_all4"],O["wrong_any"],O["wrong_none"],"vote",O["vote_macro_f1"])
# ---- slices on SVM
d=te.copy(); d["p"]=P["Linear SVM C=0.3"]; d["y"]=yte; d["err"]=(d.p!=d.y)
d["nw"]=d.text.str.split().str.len()
def sl(name,mask):
    g=d[mask]; 
    if len(g)==0: return
    fk=g[g.y==1]; rl=g[g.y==0]
    return dict(group=name,n=int(len(g)),err=float(g.err.mean()),fake_recall=float((fk.p==1).mean()) if len(fk) else None,n_fake=int(len(fk)),fpr=float((rl.p==1).mean()) if len(rl) else None,n_real=int(len(rl)))
S=[]
for nm,mk in [("<=10 words",d.nw<=10),("11-17 words",(d.nw>10)&(d.nw<=17)),("18-25 words",(d.nw>17)&(d.nw<=25)),(">25 words",d.nw>25),
  ("contains a digit",d.text.str.contains(r"\d")),("no digit",~d.text.str.contains(r"\d")),
  ("contains percent/%",d.text.str.contains(r"percent|%",case=False)),
  ("contains negation (not/no/never/n't)",d.text.str.contains(r"\b(?:not|no|never)\b|n't",case=False)),
  ("starts with 'Says'",d.text.str.match(r"^Says\b")),("does not start with 'Says'",~d.text.str.match(r"^Says\b")),
  ("mentions Obama/Clinton/Trump/Romney/Bush",d.text.str.contains(r"Obama|Clinton|Trump|Romney|Bush",case=True)),
  ("speaker is not a person (chain email, post, ad...)",d.speaker.str.contains(r"chain-email|facebook-posts|viral|tweets|blog|ad\b",case=False)),
  ("party: republican",d.party=="republican"),("party: democrat",d.party=="democrat"),("party: none",d.party=="none")]:
    S.append(sl(nm,mk))
for subj in d.subject.str.split(",").str[0].value_counts().head(6).index: S.append(sl("subject: "+subj,d.subject.str.split(",").str[0]==subj))
O["slices"]=S
for s in S: print(f"{s['group']:50s} n={s['n']:4d} err={s['err']:.3f} fakeR={s['fake_recall'] if s['fake_recall'] is None else round(s['fake_recall'],3)} fpr={s['fpr'] if s['fpr'] is None else round(s['fpr'],3)}")
# ---- threshold tuning
def tune(name):
    m=M[name]; sv=score(m,va.text); st=score(m,te.text)
    grid=np.quantile(sv,np.linspace(.2,.8,61)); best=max(grid,key=lambda t:f1_score(yva,(sv>=t).astype(int),average="macro"))
    p0=m.predict(te.text); p1=(st>=best).astype(int)
    df_=[f1_score(yte[i],p1[i],average="macro")-f1_score(yte[i],p0[i],average="macro") for i in idx]
    pr,rc,_,_=precision_recall_fscore_support(yte,p1,labels=[1],zero_division=0)
    return dict(thr=float(best),val_f1_default=f1_score(yva,m.predict(va.text),average="macro"),val_f1_tuned=f1_score(yva,(sv>=best).astype(int),average="macro"),
      test_f1_default=f1_score(yte,p0,average="macro"),test_f1_tuned=f1_score(yte,p1,average="macro"),test_fake_recall_default=float(recall_score(yte,p0)) if False else float(((p0==1)&(yte==1)).sum()/yte.sum()),
      test_fake_recall_tuned=float(rc[0]),test_fake_prec_tuned=float(pr[0]),diff=float(np.mean(df_)),diff_ci=[float(np.percentile(df_,2.5)),float(np.percentile(df_,97.5))])
O["threshold"]={k:tune(k) for k in ["LR bigram C=3","Linear SVM C=0.3","Complement NB a=1"]}
for k,v in O["threshold"].items(): print("thr",k,{a:round(b,4) if isinstance(b,float) else b for a,b in v.items()})
# ---- label mapping alternatives
def mapping(name,fakeset,drop=()):
    a=tr[~tr.label.isin(drop)];b=va[~va.label.isin(drop)];c=te[~te.label.isin(drop)]
    ya=a.label.isin(fakeset).astype(int).values;yb=b.label.isin(fakeset).astype(int).values;yc=c.label.isin(fakeset).astype(int).values
    m=Pipeline([("v",V2()),("c",LinearSVC(C=0.3,random_state=42))]).fit(a.text,ya)
    mj=Pipeline([("v",V2()),("c",DummyClassifier(strategy="most_frequent"))]).fit(a.text,ya)
    return dict(name=name,n=[len(a),len(b),len(c)],fake_share_test=float(yc.mean()),val_f1=f1_score(yb,m.predict(b.text),average="macro"),test_f1=f1_score(yc,m.predict(c.text),average="macro"),test_acc=accuracy_score(yc,m.predict(c.text)),majority_test_f1=f1_score(yc,mj.predict(c.text),average="macro"))
F3={"pants-fire","false","barely-true"}
O["mapping"]=[mapping("A current: fake = pants-fire, false, barely-true",F3),mapping("B: barely-true moved to real",{"pants-fire","false"}),mapping("C: barely-true removed",{"pants-fire","false"},drop=("barely-true",)),mapping("D: only extremes (drop barely-true and mostly-true? no: drop half-true and barely-true)",{"pants-fire","false"},drop=("half-true","barely-true"))]
for r in O["mapping"]: print(r)
# ---- 5-fold CV on train+val
al=pd.concat([tr,va]); ya=(al.bin=="fake").astype(int).values
cv=StratifiedKFold(5,shuffle=True,random_state=42); O["cv"]={}
for k in ["LR unigram","LR bigram C=3","Linear SVM C=0.3","Complement NB a=1"]:
    from sklearn.base import clone
    sc=cross_val_score(clone(M[k]),al.text,ya,cv=cv,scoring="f1_macro"); O["cv"][k]=dict(mean=float(sc.mean()),sd=float(sc.std(ddof=1)),folds=[round(float(x),4) for x in sc])
    print("cv",k,round(sc.mean(),4),round(sc.std(ddof=1),4))
json.dump(O,open("docs/week4/evaluation_results.json","w"),indent=1,default=str)
d[["label","p","y","speaker","party","subject","text"]].to_pickle("data/processed/te_svm.pkl")
