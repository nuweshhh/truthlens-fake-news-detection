import pandas as pd, numpy as np, re, html, json, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, chi2_contingency
cols=["idx","id","label","statement","subject","speaker","job","state","party","c_bt","c_f","c_ht","c_mt","c_pf","context","justification"]
dfs=[]
for s in ["train2","val2","test2"]:
    d=pd.read_csv(f"data/raw/liar/{s}.tsv",sep="\t",header=None,names=cols,quoting=3,dtype=str,keep_default_na=False)
    d["split"]=s[:-1]; dfs.append(d)
df=pd.concat(dfs,ignore_index=True)
out0={"raw_rows":len(df),"blank_rows":int((df.label=="").sum())}
df=df[df.label!=""].reset_index(drop=True)
out=dict(out0) if False else {}
out.update(out0)
out["rows"]=df.groupby("split").size().to_dict(); out["total"]=len(df)
out["labels"]=df.label.value_counts().to_dict()
fake={"pants-fire","false","barely-true"}
df["bin"]=np.where(df.label.isin(fake),"fake","real")
out["bin"]=df.bin.value_counts().to_dict()
for c in ["job","state","party","context","subject","speaker"]:
    out["empty_"+c]=int((df[c].str.strip()=="").sum())
out["empty_stmt"]=int((df.statement.str.strip()=="").sum())
out["dup_stmt"]=int(df.statement.duplicated().sum())
out["dup_cross"]=int(df[df.split!="train"].statement.isin(set(df[df.split=="train"].statement)).sum())
tok=lambda t:re.findall(r"[a-z0-9]+(?:'[a-z]+)?",t.lower())
df["n"]=df.statement.map(lambda t:len(tok(t)))
df["chars"]=df.statement.str.len()
out["len"]=df.n.describe().round(2).to_dict()
out["len_by_bin"]=df.groupby("bin").n.describe().round(2).to_dict("index")
a=df[df.bin=="real"].n; b=df[df.bin=="fake"].n
u,p=mannwhitneyu(a,b); out["mwu_p"]=float(p); out["rank_biserial"]=float(2*u/(len(a)*len(b))-1)
q1,q3=df.n.quantile([.25,.75]); iqr=q3-q1
out["iqr_bounds"]=[float(q1-1.5*iqr),float(q3+1.5*iqr)]
out["outliers_hi"]=int((df.n>q3+1.5*iqr).sum()); out["p99"]=float(df.n.quantile(.99)); out["max"]=int(df.n.max()); out["min"]=int(df.n.min())
out["short_lt5"]=int((df.n<5).sum())
# style
df["excl"]=df.statement.str.count("!"); df["caps"]=df.statement.map(lambda t:sum(c.isupper() for c in t)/max(1,len(t)))
df["digits"]=df.statement.map(lambda t:sum(c.isdigit() for c in t)/max(1,len(t)))
for f in ["excl","caps","digits"]:
    x=df[df.bin=="real"][f];y=df[df.bin=="fake"][f]
    out["style_"+f]={"real_mean":round(x.mean(),4),"fake_mean":round(y.mean(),4),"p":float(mannwhitneyu(x,y)[1])}
# html/noise
out["has_html_entity"]=int(df.statement.str.contains(r"&\w+;|<[^>]+>").sum())
out["has_url"]=int(df.statement.str.contains(r"https?://|www\.").sum())
out["non_ascii"]=int(df.statement.map(lambda t:any(ord(c)>127 for c in t)).sum())
out["has_double_space"]=int(df.statement.str.contains(r"\s{2,}").sum())
# vocab, zipf
cnt=collections.Counter(w for t in df.statement for w in tok(t))
tot=sum(cnt.values()); out["tokens"]=tot; out["vocab"]=len(cnt); out["ttr"]=round(len(cnt)/tot,4)
out["hapax_share"]=round(sum(1 for v in cnt.values() if v==1)/len(cnt),4)
# log-odds
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as SW
cr=collections.Counter(w for t in df[df.bin=="real"].statement for w in tok(t))
cf=collections.Counter(w for t in df[df.bin=="fake"].statement for w in tok(t))
nr,nf=sum(cr.values()),sum(cf.values()); V=len(cnt)
lo={w:np.log((cr[w]+.5)/(nr+.5*V))-np.log((cf[w]+.5)/(nf+.5*V)) for w in cnt if cnt[w]>=30}
srt=sorted(lo.items(),key=lambda x:x[1])
out["top_fake"]=[(w,round(v,2),cf[w],cr[w]) for w,v in srt[:12]]
out["top_real"]=[(w,round(v,2),cr[w],cf[w]) for w,v in srt[::-1][:12]]
out["stop_removed_share"]=round(sum(v for w,v in cnt.items() if w in SW)/tot,4)
out["top_words_raw"]=cnt.most_common(10)
out["negations"]={w:cnt[w] for w in ["not","no","never","nothing"]}
# metadata vs label
for c in ["party","subject","speaker"]:
    top=df[c].value_counts().head(15).index
    ct=pd.crosstab(df[df[c].isin(top)][c],df[df[c].isin(top)]["bin"])
    chi,pp,_,_=chi2_contingency(ct); k=ct.values.sum()
    out["chi_"+c]={"chi2":round(chi,1),"p":float(pp),"cramers_v":round(float(np.sqrt(chi/(k*(min(ct.shape)-1)))),3)}
# credit history leakage check
cc=["c_bt","c_f","c_ht","c_mt","c_pf"]
for c in cc: df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0)
mp={"barely-true":"c_bt","false":"c_f","half-true":"c_ht","mostly-true":"c_mt","pants-fire":"c_pf"}
sub=df[df.label.isin(mp)]
out["credit_zero_rows"]=int((df[cc].sum(axis=1)==0).sum())
# does count of own label >=1 always? (includes current statement)
ok=sum(1 for _,r in sub.iterrows() if r[mp[r.label]]>=1); out["credit_own_label_ge1"]=[ok,len(sub)]
# cleaning demo on real rows
def clean(t):
    t=html.unescape(t); t=re.sub(r"<[^>]+>"," ",t); t=re.sub(r"https?://\S+|www\.\S+"," ",t)
    t=re.sub(r"([!?])\1+",r"\1",t); t=re.sub(r"\.{2,}","...",t); return re.sub(r"\s+"," ",t).strip()
ch=df[df.statement.map(clean)!=df.statement]; out["rows_changed_by_clean"]=len(ch)
out["clean_examples"]=[(r.statement[:110],clean(r.statement)[:110]) for _,r in ch.head(3).iterrows()]
st=lambda t:[w for w in tok(t) if w not in SW]
ex=df[df.statement.str.contains(r"\bnot\b|\bnever\b",case=False)].statement.iloc[2]
out["ellipsis_rows"]=int(df.statement.str.contains(r"\.{2,}").sum())
out["ellipsis_ex"]=df[df.statement.str.contains(r"\.{2,}")].statement.iloc[0][:160]
out["curly_quotes"]=int(df.statement.str.contains("[\u2018\u2019\u201c\u201d]").sum())
out["upper_ratio_gt_20"]=int((df.caps>0.2).sum())
out["stop_example"]=(ex,st(ex))
import os; os.makedirs("data/processed",exist_ok=True)
df.to_pickle("data/processed/liar.pkl")
json.dump(out,open("docs/week2_liar_eda_summary.json","w"),indent=1,default=str)
# charts
import os; os.makedirs("docs/figures",exist_ok=True)
C=["#2a6f97","#c4553a"]
fig,ax=plt.subplots(1,2,figsize=(10,3.8))
order=["pants-fire","false","barely-true","half-true","mostly-true","true"]
v=df.label.value_counts().reindex(order)
ax[0].bar(order,v.values,color=[C[1]]*3+[C[0]]*3);ax[0].set_title("LIAR six-way labels");ax[0].tick_params(axis="x",rotation=35)
for i,x in enumerate(v.values): ax[0].text(i,x+60,str(x),ha="center",fontsize=8)
b=df.bin.value_counts();ax[1].bar(b.index,b.values,color=[C[0] if i=="real" else C[1] for i in b.index]);ax[1].set_title("Mapped to binary")
for i,x in enumerate(b.values): ax[1].text(i,x+60,str(x),ha="center")
plt.tight_layout();plt.savefig("docs/figures/fig1_labels.png",dpi=160);plt.close()
fig,ax=plt.subplots(1,2,figsize=(10,3.8))
bins=np.arange(0,df.n.max()+5,4)
for k,c in zip(["real","fake"],C): ax[0].hist(df[df.bin==k].n,bins=bins,alpha=.6,label=k,color=c)
ax[0].set_xlabel("words per statement");ax[0].set_title("Length by class");ax[0].legend()
ax[1].boxplot([df[df.bin=="real"].n,df[df.bin=="fake"].n],tick_labels=["real","fake"]);ax[1].set_title("Boxplot (outliers as dots)")
plt.tight_layout();plt.savefig("docs/figures/fig2_length.png",dpi=160);plt.close()
fig,ax=plt.subplots(1,2,figsize=(10,3.8))
r=np.array(sorted(cnt.values(),reverse=True));ax[0].loglog(np.arange(1,len(r)+1),r,color=C[0]);ax[0].set_title("Rank-frequency (Zipf) plot");ax[0].set_xlabel("rank");ax[0].set_ylabel("count")
tf=out["top_fake"][:10][::-1];tr=out["top_real"][:10][::-1]
ax[1].barh([w for w,_,_,_ in tr]+[w for w,_,_,_ in tf],[v for _,v,_,_ in tr]+[v for _,v,_,_ in tf],color=[C[0]]*len(tr)+[C[1]]*len(tf));ax[1].set_title("Log-odds: real (blue) vs fake (red)");ax[1].tick_params(axis="y",labelsize=8)
plt.tight_layout();plt.savefig("docs/figures/fig3_vocab.png",dpi=160);plt.close()
fig,ax=plt.subplots(figsize=(6,3.8))
e=pd.Series({c:int((df[c].str.strip()=="").sum()) for c in ["job","state","party","context","subject","speaker"]})
ax.bar(e.index,e.values,color=C[1]);ax.set_title("Empty fields in LIAR (all splits)")
for i,x in enumerate(e.values): ax.text(i,x+20,str(x),ha="center",fontsize=8)
plt.tight_layout();plt.savefig("docs/figures/fig4_missing.png",dpi=160);plt.close()

