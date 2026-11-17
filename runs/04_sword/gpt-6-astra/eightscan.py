import numpy as np,itertools,time
X=np.load('analysis_outputs/X.npy');n=350;d=30
f=open('analysis_outputs/log.txt','a',buffering=1)
def log(s):print(s,flush=True);f.write(str(s)+'\n')
N=27405*8;K=np.zeros((N,d),np.int8);P=np.empty((n,N),np.complex64);Z=np.exp(2j*np.pi*X[:n]).astype(np.complex64)
log('Eight-column Fourier search: all four-column signed-sum features, then pair comparisons. Screen 350 rows at amplitude >0.24; validate on separate 3000 rows.')
c=0
for inds in itertools.combinations(range(d),4):
 for signs in itertools.product([-1,1],repeat=3):
  K[c,list(inds)]=[1,*signs];P[:,c]=np.prod(Z[:,inds]**np.array([1,*signs])[None,:],axis=1);c+=1
log(f'Eight scan features {N}; memory {P.nbytes/1e6:.0f} MB');t=time.time();res=[]
for start in range(0,N,128):
 for conj in [False,True]:
  A=(P[:,start:start+128].conj().T if conj else P[:,start:start+128].T)@P[:,start:]/n
  ii,jj=np.where(abs(A)>.24)
  for i,j in zip(ii,jj):
   k=K[start+j]+(-1 if conj else 1)*K[start+i]
   if np.any(k):res.append((float(abs(A[i,j])),k.tolist()))
 if start%8192==0:log(f'Eight: {start}/{N}; candidates {len(res)}; elapsed {time.time()-t:.1f}s')
seen=set();out=[]
for v,k in sorted(res,reverse=True):
 k=np.array(k);k*=np.sign(k[np.flatnonzero(k)[0]]);key=tuple(k)
 if key in seen:continue
 seen.add(key);val=abs(np.exp(2j*np.pi*(X[2000:5000]@k)).mean());out.append((val,v,k.tolist()))
out.sort(reverse=True)
for val,v,k in out[:30]:log(f'Eight validation {val:.5f}, screen {v:.5f}: {[(i+1,a) for i,a in enumerate(k) if a]}')
np.savez('analysis_outputs/eight.npz',K=[k for val,v,k in out],values=[val for val,v,k in out],screen=[v for val,v,k in out])
log('Eight-column search complete: unique screened candidates '+str(len(out)))
