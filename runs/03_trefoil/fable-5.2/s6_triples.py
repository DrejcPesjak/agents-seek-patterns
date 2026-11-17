import numpy as np, pandas as pd, itertools, time
X=pd.read_csv('X.csv').values; n,d=X.shape
K=8; ks=np.array([k for k in range(-K,K+1) if k])
E=np.exp(2j*np.pi*X[:,:,None]*ks[None,None,:]).astype(np.complex64)  # n,d,2K
best=[];t0=time.time()
for i in range(d):
    for j in range(i+1,d):
        P=(E[:,i,K:,None]*E[:,j,None,:]).reshape(n,-1)        # k1>0, k2 any: n x K*2K
        rest=E[:,j+1:,:].reshape(n,-1)
        if rest.shape[1]==0: continue
        G=np.abs(P.T@rest)**2/n
        for t in np.argsort(G.ravel())[::-1][:2]:
            a,b=np.unravel_index(t,G.shape)
            best.append((float(G[a,b]),(i+1,j+1,j+2+b//(2*K)),(int(ks[K+a//(2*K)]),int(ks[a%(2*K)]),int(ks[b%(2*K)]))))
best.sort(reverse=True)
N=4060*K*(2*K)**2
print('triples K=8 tests',N,'exp max',np.log(N),'time',time.time()-t0)
for b in best[:10]: print('  ',round(b[0],1),b[1],b[2])
