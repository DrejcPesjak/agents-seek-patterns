import numpy as np, pandas as pd, itertools, time, cupy as cp
X=pd.read_csv('X.csv').values; n,d=X.shape
def cols(K,r):
    kk=[k for k in range(-K,K+1) if k]
    idxs=list(itertools.combinations(range(d),r)); cos=np.array(list(itertools.product(kk,repeat=r)),np.float32)
    A=np.zeros((len(idxs)*len(cos),d),np.float32)
    for t,idx in enumerate(idxs): A[t*len(cos):(t+1)*len(cos),list(idx)]=cos
    return A
def scan(A,B,m,name,chunk=4096,thr=30,seed=0,weight=None):
    rng=np.random.default_rng(seed); rows=rng.choice(n,m,replace=False)
    Xg=cp.asarray(X[rows],dtype=cp.float32)
    Bg=cp.asarray(B); EB=cp.exp(2j*cp.pi*(Xg@Bg.T)).astype(cp.complex64)
    if weight is not None: EB=EB*cp.asarray(weight[rows],dtype=cp.complex64)[:,None]
    sB=(cp.abs(Bg)>0).astype(cp.float32)
    hits=[];t0=time.time(); gmax=0
    for s in range(0,len(A),chunk):
        Ag=cp.asarray(A[s:s+chunk]); EA=cp.exp(2j*cp.pi*(Xg@Ag.T)).astype(cp.complex64)
        G=cp.abs(EA.T@EB)**2/m
        G[((cp.abs(Ag)>0).astype(cp.float32)@sB.T)>0]=0
        gmax=max(gmax,float(G.max()))
        ii,jj=cp.nonzero(G>thr)
        if len(ii):
            v=G[ii,jj].get()
            for i,j,val in zip(ii.get(),jj.get(),v): hits.append((float(val),A[s+i]+B[j]))
    print(name,'rows',m,'max stat',round(gmax,1),'time',round(time.time()-t0),'s, hits>',thr,':',len(hits))
    seen={}
    for val,v in sorted(hits,key=lambda t:-t[0]):
        if v[np.nonzero(v)[0][0]]<0: v=-v
        key=tuple(v.astype(int))
        if key not in seen: seen[key]=val
    for key,val in list(seen.items())[:40]:
        print('  ',round(val,1),{i+1:c for i,c in enumerate(key) if c})
    return seen
if __name__=='__main__':
    P2=cols(2,2); T2=cols(2,3); print(P2.shape,T2.shape)
    scan(T2,P2,8000,'5-col coef<=2',thr=30)
    scan(T2,T2,1500,'6-col coef<=2',thr=40)
