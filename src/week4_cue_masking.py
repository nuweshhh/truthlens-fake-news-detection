import pandas as pd, numpy as np, re, json, warnings
warnings.filterwarnings("ignore")
import os; exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"week3_validation_experiments.py")).read().split("R=[]")[0])
rng=np.random.default_rng(42); idx=[rng.integers(0,len(yte),len(yte)) for _ in range(2000)]
def mk(): return Pipeline([("v",vec(ngram_range=(1,2))),("c",LinearSVC(C=0.3,random_state=42))])
def nosays(t): return re.sub(r"^Says\s+","",t)
def nonum(t): return re.sub(r"\d[\d,\.]*","NUM",t)
def noname(t):
    toks=t.split(); out=[]
    for i,w in enumerate(toks):
        core=re.sub(r"[^\w']","",w)
        out.append("NAME" if (core[:1].isupper() and i>0 and core.lower() not in {"i","u.s","americans","american"} and toks[i-1][-1:] not in ".?!") else w)
    return " ".join(out)
V={"default":lambda t:t,"remove leading 'Says'":nosays,"mask numbers (NUM)":nonum,"mask capitalised names (NAME)":noname,
   "mask names and numbers":lambda t:nonum(noname(t)),"all three":lambda t:nonum(noname(nosays(t)))}
R={};P={}
for k,f in V.items():
    m=mk().fit(tr.text.map(f),ytr); pv=m.predict(va.text.map(f)); pt=m.predict(te.text.map(f)); P[k]=pt
    R[k]=dict(val=f1_score(yva,pv,average="macro"),test=f1_score(yte,pt,average="macro"),test_acc=accuracy_score(yte,pt),fake_recall=float(((pt==1)&(yte==1)).sum()/yte.sum()))
for k in V:
    if k=="default": continue
    d=[f1_score(yte[i],P[k][i],average="macro")-f1_score(yte[i],P["default"][i],average="macro") for i in idx]
    R[k]["diff"]=float(np.mean(d)); R[k]["ci"]=[float(np.percentile(d,2.5)),float(np.percentile(d,97.5))]
for k,v in R.items(): print(f"{k:32s}",{a:(round(b,4) if isinstance(b,float) else [round(x,4) for x in b]) for a,b in v.items()})
print(noname("Says U.S. Rep. Connie Mack IV passed only one bill in seven years.")); print(noname("Barack Obama has provided guns to Mexican drug cartels."))
json.dump(R,open("docs/week4/cue_masking_results.json","w"),indent=1)
