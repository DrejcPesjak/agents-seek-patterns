import numpy as np, pandas as pd, itertools, time
X=pd.read_csv('X.csv').values; n,d=X.shape
def top(S,lab,k=8):
    idx=np.argsort(S.ravel())[::-1][:k]
    for t in idx: print('  ',round(float(S.ravel()[t]),1),lab(np.unravel_index(t,S.shape)))
# --- pairs, high freq K=40
K=40; ks=[k for k in range(-K,K+1) if k]
E=np.exp(2j*np.pi*X[:,:,None]*np.array(ks)[None,None,:]).reshape(n,-1).astype(np.complex64) # n x d*2K
G=np.abs(E.conj().T@E)**2/n
c=np.arange(d).repeat(len(ks)); G[c[:,None]==c[None,:]]=0
print('pairs K=40: tests',G.size/4,'exp max',np.log(G.size/4))
top(G,lambda ij:((c[ij[0]]+1,-ks[ij[0]%len(ks)]),(c[ij[1]]+1,ks[ij[1]%len(ks)])))
del G,E
# --- quads via pair x pair, coefs up to 2
def cols(K,r):
    L=[];V=[]
    kk=[k for k in range(-K,K+1) if k]
    for idx in itertools.combinations(range(d),r):
        for co in itertools.product(kk,repeat=r):
            L.append((idx,co)); 
    return L
def build(L):
    A=np.zeros((len(L),d),np.float32)
    for t,(idx,co) in enumerate(L): A[t,list(idx)]=co
    return A
def scan(LA,LB,name,chunk=2000):
    A=build(LA);B=build(LB)
    EB=np.exp(2j*np.pi*(X@B.T)).astype(np.complex64)
    best=[];t0=time.time()
    for s in range(0,len(LA),chunk):
        EA=np.exp(2j*np.pi*(X@A[s:s+chunk].T)).astype(np.complex64)
        G=np.abs(EA.T@EB)**2/n
        # zero-out where combined vector has support smaller than sum (overlap)
        ov=(np.abs(A[s:s+chunk])>0).astype(np.float32)@(np.abs(B)>0).astype(np.float32).T
        G[ov>0]=0
        idx=np.argsort(G.ravel())[::-1][:5]
        for t in idx:
            i,j=np.unravel_index(t,G.shape); best.append((float(G[i,j]),LA[s+i],LB[j]))
    best.sort(key=lambda r:-r[0])
    print(name,'time',round(time.time()-t0),'s; top:')
    seen=set()
    for b in best:
        v=A[0]*0
        for (idx,co) in b[1:]: v[list(idx)]+=co
        key=tuple(v) if v[np.nonzero(v)[0][0]]>0 else tuple(-v)
        if key in seen: continue
        seen.add(key); print('  ',round(b[0],1),{int(i)+1:int(v[i]) for i in np.nonzero(v)[0]})
        if len(seen)>=8: break
P2=cols(2,2); P1=cols(1,2); T1=cols(1,3)
print('quads coef<=2: approx tests',27405*128,'exp max',np.log(27405*128)); scan(P2,P2,'quads')
print('5-col ±1: tests',142506*16,'exp max',np.log(142506*16)); scan(T1,P1,'5col')
print('6-col ±1: tests',593775*32,'exp max',np.log(593775*32)); scan(T1,T1,'6col')
