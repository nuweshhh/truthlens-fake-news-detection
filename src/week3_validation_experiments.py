import pandas as pd, numpy as np, re, json, time, warnings
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, roc_auc_score
from nltk.stem import PorterStemmer
import html
df=pd.read_pickle("data/processed/liar.pkl")
def clean(t):
    t=html.unescape(t); t=re.sub(r"<[^>]+>"," ",t); t=re.sub(r"https?://\S+|www\.\S+"," ",t)
    t=re.sub(r"([!?])\1+",r"\1",t); t=re.sub(r"\.{2,}","...",t); return re.sub(r"\s+"," ",t).strip()
df["text"]=df.statement.map(clean)
tr=df[df.split=="train"].copy(); va=df[df.split=="val"].copy(); te=df[df.split=="test"].copy()
trs=set(tr.text)
n0=(len(va),len(te)); va=va[~va.text.isin(trs)]; te=te[~te.text.isin(trs)]
info={"dropped_overlap":[n0[0]-len(va),n0[1]-len(te)],"n":[len(tr),len(va),len(te)]}
y=lambda d:(d.bin=="fake").astype(int).values   # 1 = fake
ytr,yva,yte=y(tr),y(va),y(te)
ps=PorterStemmer(); cache={}
def stem_tok(t):
    return [cache.setdefault(w,ps.stem(w)) for w in re.findall(r"[a-z0-9]+(?:'[a-z]+)?",t.lower())]
SW_KEEP_NEG=sorted(set(ENGLISH_STOP_WORDS)-{"not","no","never","nothing","cannot","without","against","none","nor"})
def vec(**kw): 
    base=dict(sublinear_tf=True,min_df=2)
    base.update(kw); return TfidfVectorizer(**base)
def ev(model,Xa,ya,Xb,yb):
    model.fit(Xa,ya); p=model.predict(Xb)
    s=model.decision_function(Xb) if hasattr(model,"decision_function") else model.predict_proba(Xb)[:,1]
    pr,rc,f,_=precision_recall_fscore_support(yb,p,labels=[1],zero_division=0)
    return dict(macro_f1=f1_score(yb,p,average="macro"),acc=accuracy_score(yb,p),fake_p=pr[0],fake_r=rc[0],auc=roc_auc_score(yb,s)),p
R=[]
def run(name,family,pipe,texts=("text",)):
    t0=time.time(); m,_=ev(pipe,tr.text,ytr,va.text,yva); m.update(name=name,family=family,secs=round(time.time()-t0,2)); R.append(m); print(f"{name:55s} F1={m['macro_f1']:.4f} acc={m['acc']:.4f}")
# majority
from sklearn.dummy import DummyClassifier
run("Majority class","baseline",Pipeline([("v",vec()),("c",DummyClassifier(strategy="most_frequent"))]))
LR=lambda C=1,**k:LogisticRegression(C=C,max_iter=2000,solver="liblinear",random_state=42,**k)
run("LR, unigram TF-IDF (default)","LR",Pipeline([("v",vec()),("c",LR())]))
run("LR, uni+bigram","LR",Pipeline([("v",vec(ngram_range=(1,2))),("c",LR())]))
run("LR, unigram, sklearn stop words removed","LR",Pipeline([("v",vec(stop_words="english")),("c",LR())]))
run("LR, unigram, stop words removed but negations kept","LR",Pipeline([("v",vec(stop_words=SW_KEEP_NEG)),("c",LR())]))
run("LR, unigram, Porter stemming","LR",Pipeline([("v",vec(tokenizer=stem_tok,token_pattern=None,lowercase=False)),("c",LR())]))
run("LR, unigram, no lowercasing","LR",Pipeline([("v",vec(lowercase=False)),("c",LR())]))
run("LR, char 2-5 grams","LR",Pipeline([("v",vec(analyzer="char_wb",ngram_range=(2,5))),("c",LR())]))
run("LR, unigram, class_weight=balanced","LR",Pipeline([("v",vec()),("c",LR(class_weight="balanced"))]))
for C in [0.1,0.3,1,3,10]:
    run(f"LR, uni+bigram, C={C}","LR-C",Pipeline([("v",vec(ngram_range=(1,2))),("c",LR(C))]))
for C in [0.03,0.1,0.3,1]:
    run(f"Linear SVM, uni+bigram, C={C}","SVM",Pipeline([("v",vec(ngram_range=(1,2))),("c",LinearSVC(C=C,random_state=42))]))
for a in [0.1,0.3,1.0]:
    run(f"Multinomial NB, uni+bigram, alpha={a}","NB",Pipeline([("v",vec(ngram_range=(1,2))),("c",MultinomialNB(alpha=a))]))
    run(f"Complement NB, uni+bigram, alpha={a}","NB",Pipeline([("v",vec(ngram_range=(1,2))),("c",ComplementNB(alpha=a))]))
for k in [1000,5000,20000]:
    run(f"LR, uni+bigram, chi2 top-{k}","FS",Pipeline([("v",vec(ngram_range=(1,2))),("s",SelectKBest(chi2,k=k)),("c",LR(1))]))
json.dump(R,open("docs/week3/validation_results.json","w"),indent=1)
vr=pd.DataFrame(R); print(vr[["name","macro_f1"]].sort_values("macro_f1",ascending=False).head(6).to_string())
# --- shortcut experiments (metadata)
from scipy.sparse import hstack, csr_matrix
def meta_text(d): return ("spk_"+d.speaker.str.replace(r"\W","_",regex=True)+" party_"+d.party.str.replace(r"\W","_",regex=True)+" subj_"+d.subject.str.split(",").str[0].str.replace(r"\W","_",regex=True))
S={}
m,_=ev(Pipeline([("v",vec(min_df=1,token_pattern=r"\S+")),("c",LR())]),meta_text(tr),ytr,meta_text(va),yva); S["speaker+party+subject only"]=m
cc=["c_bt","c_f","c_ht","c_mt","c_pf"]
Xc=lambda d:d[cc].astype(float).values
from sklearn.preprocessing import StandardScaler
sc=StandardScaler().fit(Xc(tr))
m,_=ev(LR(),sc.transform(Xc(tr)),ytr,sc.transform(Xc(va)),yva); S["credit-history counts only"]=m
v=vec(ngram_range=(1,2)); Xt=v.fit_transform(tr.text); Xv=v.transform(va.text)
m,_=ev(LR(),hstack([Xt,csr_matrix(sc.transform(Xc(tr)))]).tocsr(),ytr,hstack([Xv,csr_matrix(sc.transform(Xc(va)))]).tocsr(),yva); S["text + credit-history counts"]=m
json.dump(S,open("docs/week3/shortcut_results.json","w"),indent=1)
for k,v_ in S.items(): print(k,round(v_["macro_f1"],4),round(v_["acc"],4))
# --- final on test
best={"LR":Pipeline([("v",vec(ngram_range=(1,2))),("c",LR(1))])}
F={}; preds={}
cands={"Majority class":Pipeline([("v",vec()),("c",DummyClassifier(strategy="most_frequent"))]),
 "LR unigram (default)":Pipeline([("v",vec()),("c",LR())])}
def pick(fam,key=None):
    d=vr[vr.family.isin(fam)].sort_values("macro_f1",ascending=False).iloc[0]; return d.name_ if False else d["name"]
print("val-best per family:",{f:pick([f]) for f in ["LR-C","SVM","NB","FS"]})
json.dump({"info":info},open("docs/week3/split_info.json","w"))
tr.to_pickle("data/processed/tr.pkl");va.to_pickle("data/processed/va.pkl");te.to_pickle("data/processed/te.pkl")
