import numpy as np, pandas as pd, itertools, struct
X=pd.read_csv('../X.csv').values; n,d=X.shape
# low bits
b=X.view(np.uint64); print('low bit mean per col',np.round((b&1).mean(0),2)[:10], 'trailing zeros avg',np.mean([(int(v)&-int(v)).bit_length()-1 for v in b[:500].ravel() if v]))
# single high freq
ks=np.arange(1,3001)
best=[]
for i in range(d):
    v=np.abs(np.exp(2j*np.pi*np.outer(X[:,i],ks)).mean(0)); best.append((v.max(),i+1,ks[v.argmax()]))
print('single hf',sorted(best,reverse=True)[:5],'expected max ~',np.sqrt(np.log(3000*30)/n))
# pairs high K
K=48; kk=[k for k in range(-K,K+1) if k]
F=np.concatenate([np.exp(2j*np.pi*np.outer(X[:,i],kk)) for i in range(d)],1).astype(np.complex64)
G=np.abs(F.T@F)/n
m=len(kk)
for i in range(d): G[i*m:(i+1)*m,i*m:(i+1)*m]=0
idx=np.argsort(G.ravel())[::-1][:12:2]
for t in idx:
    p,q=divmod(t,G.shape[1]); print('pair %.4f'%G[p,q],(p//m+1,kk[p%m]),(q//m+1,kk[q%m]))
print('expected noise max ~',np.sqrt(np.log(G.size/2)/n))
# triples K=6
K=6; kk=[k for k in range(-K,K+1) if k]; pos=[k for k in kk if k>0]
S=[np.exp(2j*np.pi*np.outer(X[:,i],kk)).astype(np.complex64) for i in range(d)]
best=[]
for i,j in itertools.combinations(range(d),2):
    P=(S[i][:,K:,None]*S[j][:,None,:]).reshape(n,-1)  # a>0, b any
    for k in range(j+1,d):
        G=np.abs(P.T@S[k])/n
        t=G.argmax(); p,q=divmod(t,len(kk))
        best.append((G.max(),(i+1,pos[p//len(kk)]),(j+1,kk[p%len(kk)]),(k+1,kk[q])))
best.sort(reverse=True); print('triples K6',best[:6]); print('exp noise max',np.sqrt(np.log(4060*6*12*12)/n))
