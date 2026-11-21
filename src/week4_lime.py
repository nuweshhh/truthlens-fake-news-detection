import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings("ignore")
import os; exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"week3_validation_experiments.py")).read().split("R=[]")[0])
from lime.lime_text import LimeTextExplainer
m=Pipeline([("v",vec(ngram_range=(1,2))),("c",LogisticRegression(C=3,max_iter=2000,solver="liblinear",random_state=42))]).fit(tr.text,ytr)
te=te.reset_index(drop=True); te["p"]=m.predict(te.text); te["pf"]=m.predict_proba(te.text)[:,1]; te["y"]=yte
pick=[]
def find(sub): 
    r=te[te.text.str.contains(sub,regex=False)]; return r.index[0]
pick=[("Error: real predicted fake",find("passed only one bill in seven years")),("Error: fake predicted real",find("roughly 15 percent of Americans who dont have health insurance")),]
ok_f=te[(te.y==1)&(te.p==1)].sort_values("pf",ascending=False).index[0]; ok_r=te[(te.y==0)&(te.p==0)].sort_values("pf").index[0]
pick+=[("Correct: fake",ok_f),("Correct: real",ok_r)]
ex=LimeTextExplainer(class_names=["real","fake"],random_state=42)
out=[]
for title,i in pick:
    e=ex.explain_instance(te.text[i],m.predict_proba,num_features=8,num_samples=3000)
    out.append(dict(title=title,label=te.label[i],truth=int(te.y[i]),pred_fake_prob=round(float(te.pf[i]),3),text=te.text[i],feats=[(w,round(float(v),3)) for w,v in e.as_list()]))
    print(title,te.label[i],round(te.pf[i],3),te.text[i][:110]); print("   ",out[-1]["feats"])
json.dump(out,open("docs/week4/lime_explanations.json","w"),indent=1)
