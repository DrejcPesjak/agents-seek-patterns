import numpy as np, pandas as pd, itertools, time
X=pd.read_csv('../X.csv').values; n,d=X.shape
K=4
E={k:np.exp(2j*np.pi*k*X) for k in range(-K,K+1)}
thr=5/np.sqrt(n)
print('thr',thr)
res=[]
# size1
for i in range(d):
    for k in range(1,9):
        v=abs(np.exp(2j*np.pi*k*X[:,i]).mean())
        if v>thr: res.append((v,(i+1,),(k,)))
# size 2
for i,j in itertools.combinations(range(d),2):
    for a in range(1,K+1):
        B=np.stack([E[b][:,j] for b in range(-K,K+1) if b],1)
        v=np.abs((E[a][:,i][:,None]*B).mean(0))
        for b,vv in zip([b for b in range(-K,K+1) if b],v):
            if vv>thr: res.append((vv,(i+1,j+1),(a,b)))
print('pairs done',sorted(res,reverse=True)[:20])
K3=3; ks=[b for b in range(-K3,K3+1) if b]
t=time.time()
for i,j,k in itertools.combinations(range(d),3):
    for a in range(1,K3+1):
        A=E[a][:,i]
        Bj=np.stack([E[b][:,j] for b in ks],1)*A[:,None]   # n x 6
        Ck=np.stack([E[c][:,k] for c in ks],1)            # n x 6
        M=np.abs(Bj.T@Ck)/n
        for (bi,ci) in zip(*np.where(M>thr)):
            res.append((M[bi,ci],(i+1,j+1,k+1),(a,ks[bi],ks[ci])))
print('triples',time.time()-t)
for r in sorted(res,reverse=True)[:40]: print(r)
