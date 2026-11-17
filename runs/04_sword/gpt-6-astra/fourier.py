import numpy as np,time,itertools
X=np.load('analysis_outputs/X.npy'); X=X[:5000]; n=len(X)
f=open('analysis_outputs/log.txt','a',buffering=1)
def log(s):print(s,flush=True);f.write(str(s)+'\n')
Z=np.exp(2j*np.pi*X).astype(np.complex64)
ks=[];vs=[]
for i in range(30):
 for j in range(i+1,30):
  for s in [-1,1]:
   k=np.zeros(30,int);k[i]=1;k[j]=s;ks.append(k);vs.append(Z[:,i]*Z[:,j]**s)
P=np.array(vs).T.copy();K=np.array(ks)
for power in [1,2,3]:
 V=Z**power
 for conjugate in [False,True]:
  A=(V.conj().T if conjugate else V.T) @ P/n
  flat=np.argsort(abs(A).ravel())[-12:][::-1]
  for idx in flat[:5]:
   i,j=np.unravel_index(idx,A.shape); k=K[j].copy();k[i]+=(-1 if conjugate else 1)*power
   log(f'triple {abs(A[i,j]):.5f}: {[(q+1,int(v)) for q,v in enumerate(k) if v]}')
log('Four-way scan')
res=[]
for start in range(0,len(K),100):
 for conj in [False,True]:
  A=(P[:,start:start+100].conj().T if conj else P[:,start:start+100].T)@P/n
  flat=np.argpartition(abs(A).ravel(),-20)[-20:]
  for idx in flat:
   i,j=np.unravel_index(idx,A.shape);k=K[j]+(-1 if conj else 1)*K[start+i]
   if np.any(k):res.append((float(abs(A[i,j])),k.tolist()))
res.sort(reverse=True);seen=set();out=[]
for v,k in res:
 key=tuple(k) if next(x for x in k if x)!=-1 else tuple(-x for x in k)
 if key in seen:continue
 seen.add(key);out.append((v,k));log(f'quad {v:.5f}: {[(q+1,v) for q,v in enumerate(k) if v]}')
 if len(out)==40:break
np.savez('analysis_outputs/fourier.npz',values=[v for v,k in out],K=[k for v,k in out])
