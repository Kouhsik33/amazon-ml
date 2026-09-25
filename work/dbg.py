import sys, numpy as np, pandas as pd
sys.path.insert(0, r"D:\Dataset_ML_C\work")
from er_norm import norm_name, norm_addr
from er_block import *
W=r"D:\Dataset_ML_C\work"
S={n:pd.read_pickle(f"{W}/sample_{n}.pkl") for n in ("s1","s2","s3")}
for n,d in S.items():
    d["name_n"]=[norm_name(x) for x in d.business_name.values]; d["addr_n"]=[norm_addr(x) for x in d.business_address.values]
targets=[("Foundry Inc","S3-408780185"),("Shyam Institute of Technology Co","S3-68731259"),("Shiv Consulting","S3-109442597")]
for nm,mid in targets:
    r=S["s1"][S["s1"].business_name==nm]
    m=S["s3"][S["s3"].entity_id==mid].iloc[0]
    for _,x in r.iterrows():
        print(x.entity_id, x.name_n,"|",x.addr_n,"||", m.name_n,"|",m.addr_n, x.country==m.country)

print("----- debug scoring")
c="US"
sub={n:d[d.country==c].reset_index(drop=True) for n,d in S.items()}
allname=np.concatenate([d.name_n.values for d in sub.values()]); alladdr=np.concatenate([d.addr_n.values for d in sub.values()])
sn,sa=token_stopset(allname,0.005),token_stopset(alladdr,0.005)
print("stop addr sample:",sorted(sa)[:60])
sides={}
for n,d in sub.items():
    rn,kn=keyify(d.name_n.values,name_keys,sn); ra,ka=keyify(d.addr_n.values,addr_keys,sa)
    sides[n]=(rn,kn,ra,ka,len(d))
M=build_matrices(sides,600)
i=int(np.where(sub["s1"].entity_id=="S1-509252308")[0][0]); j=int(np.where(sub["s3"].entity_id=="S3-408780185")[0][0])
An,Aa=M["s1"]; Bn,Ba=M["s3"]
print("name score",(An[i]@Bn[j].T).sum(),"addr score",(Aa[i]@Ba[j].T).sum())
P=(An[i]@Bn.T+Aa[i]@Ba.T).toarray().ravel()
print("true score",P[j],"rank",(P>P[j]).sum(),"nnz",(P>0).sum(), "top5",np.sort(P)[-5:])
print(addr_keys(sub["s1"].addr_n[i],sa)); print(addr_keys(sub["s3"].addr_n[j],sa))
