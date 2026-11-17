import numpy as np,itertools,time
X=np.load('analysis_outputs/X.npy');n=1200
f=open('analysis_outputs/log.txt','a',buffering=1)
def log(s): print(s,flush=True);f.write(str(s)+'\n')
log('Six-variable Fourier search, screen 1200 rows, validate on 3000 separate rows.')
Z=np.exp(2j*np.pi*X[:n]).astype(np.complex64)
K=[];P=[]
for inds in itertools.combinations(range(30),3):
 for signs in itertools.product([-1,1],repeat=2):
  k=np.zeros(30,np.int8);k[list(inds)]=[1,*signs];K.append(k)
  P.append(Z[:,inds[0]]*Z[:,inds[1]]**signs[0]*Z[:,inds[2]]**signs[1])
K=np.array(K);P=np.array(P).T.copy();log(str(P.shape))
res=[];t=time.time()
for start in range(0,len(K),128):
 for conj in [False,True]:
  A=(P[:,start:start+128].conj().T if conj else P[:,start:start+128].T)@P/n
  ii,jj=np.where(abs(A)>.117)
  for i,j in zip(ii,jj):
   k=K[j]+(-1 if conj else 1)*K[start+i]
   if np.any(k):res.append((float(abs(A[i,j])),k.tolist()))
 if start%2048==0:log(f'{start}/{len(K)}; candidates {len(res)}; seconds {time.time()-t:.1f}')
seen=set();checked=[]
for v,k in sorted(res,reverse=True):
 k=np.array(k);k=k*np.sign(k[np.flatnonzero(k)[0]]);key=tuple(k)
 if key in seen:continue
 seen.add(key); val=abs(np.exp(2j*np.pi*(X[2000:5000]@k)).mean());checked.append((val,v,k.tolist()))
checked.sort(reverse=True)
for val,v,k in checked[:30]: log(f'six validation {val:.5f}, train {v:.5f}: {[(i+1,a) for i,a in enumerate(k) if a]}')
np.savez('analysis_outputs/six.npz',K=[k for val,v,k in checked],values=[val for val,v,k in checked])
