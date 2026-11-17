import numpy as np, pandas as pd, itertools, time
X=pd.read_csv('../X.csv').values; n,d=X.shape
def E(k,i): return np.exp(2j*np.pi*k*X[:,i]).astype(np.complex64)
def pairfeats(K):
    F=[];L=[]
    for i,j in itertools.combinations(range(d),2):
        for a in range(1,K+1):
            for b in [b for b in range(-K,K+1) if b]:
                F.append(E(a,i)*E(b,j));L.append(((i,a),(j,b)))
    return np.array(F).T,L
def tripfeats():
    F=[];L=[]
    for i,j,k in itertools.combinations(range(d),3):
        for b in (1,-1):
            for c in (1,-1):
                F.append(E(1,i)*E(b,j)*E(c,k));L.append(((i,1),(j,b),(k,c)))
    return np.array(F).T,L
def combine(l1,l2,s2):
    D={}
    for i,a in l1: D[i]=D.get(i,0)+a
    for i,a in l2: D[i]=D.get(i,0)+s2*a
    return {i+1:a for i,a in D.items() if a}
def gram(F1,L1,F2,L2,thr,minsup,tag):
    found={}
    bs=512
    for s in range(0,F1.shape[1],bs):
        A=F1[:,s:s+bs]
        for s2,G in ((1,np.abs(A.T@F2)/n),(-1,np.abs(A.conj().T@F2)/n)):
            # A.T@F2 = mean of f1*f2 (sum), conj -> f2 - f1
            for p,q in zip(*np.where(G>thr)):
                c=combine(L2[q],L1[s+p],s2 if s2==1 else -1)
                if len(c)>=minsup:
                    key=tuple(sorted(c.items())); 
                    if key[0][1]<0: key=tuple((i,-a) for i,a in key)
                    found[key]=max(found.get(key,0),G[p,q])
    r=sorted(found.items(),key=lambda t:-t[1])
    print(tag,'nfound',len(r),flush=True)
    for k,v in r[:15]: print('  %.4f'%v,k,flush=True)
t=time.time()
F2,L2=pairfeats(2); print(F2.shape)
gram(F2,L2,F2,L2,0.06,4,'size4 K=2'); print(time.time()-t,flush=True)
F1,L1=pairfeats(1); F3,L3=tripfeats(); print(F3.shape)
gram(F1,L1,F3,L3,0.062,5,'size5 pm1'); print(time.time()-t,flush=True)
gram(F3,L3,F3,L3,0.068,6,'size6 pm1'); print(time.time()-t,flush=True)
