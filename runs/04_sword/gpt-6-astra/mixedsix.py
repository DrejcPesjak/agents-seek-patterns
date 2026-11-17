import numpy as np,itertools,time
X=np.load('analysis_outputs/X.npy');n=600;d=30
f=open('analysis_outputs/log.txt','a',buffering=1)
def log(s):print(s,flush=True);f.write(str(s)+'\n')
N=4060*32;K=np.zeros((N,d),np.int8);P=np.empty((n,N),np.complex64);Z=np.exp(2j*np.pi*X[:n]).astype(np.complex64)
log('Six-column mixed integer coefficient search: all triple features with nonzero coefficients in [-2,2]. Screen 600 rows; independent validation rows 2000:5000.')
c=0
for inds in itertools.combinations(range(d),3):
 for a,b,v in itertools.product([1,2],[-2,-1,1,2],[-2,-1,1,2]):
  K[c,list(inds)]=[a,b,v];P[:,c]=Z[:,inds[0]]**a*Z[:,inds[1]]**b*Z[:,inds[2]]**v;c+=1
log(f'Feature count {N}, memory {P.nbytes/1e6:.0f} MB')
t=time.time();res=[]
for start in range(0,N,128):
 # upper-triangular comparison cuts duplication by half
 for conj in [False,True]:
  A=(P[:,start:start+128].conj().T if conj else P[:,start:start+128].T)@P[:,start:]/n
  ii,jj=np.where(abs(A)>.19)
  for i,j in zip(ii,jj):
   k=K[start+j]+(-1 if conj else 1)*K[start+i]
   if np.any(k):res.append((float(abs(A[i,j])),k.tolist()))
 if start%4096==0:log(f'Mixed six: {start}/{N}; candidates {len(res)}; elapsed {time.time()-t:.1f}s')
seen=set();out=[]
for v,k in sorted(res,reverse=True):
 k=np.array(k);k*=np.sign(k[np.flatnonzero(k)[0]]);key=tuple(k)
 if key in seen:continue
 seen.add(key); val=abs(np.exp(2j*np.pi*(X[2000:5000]@k)).mean());out.append((val,v,k.tolist()))
out.sort(reverse=True)
for val,v,k in out[:30]:log(f'Mixed-six validation {val:.5f}, screen {v:.5f}: {[(i+1,a) for i,a in enumerate(k) if a]}')
np.savez('analysis_outputs/mixedsix.npz',K=[k for val,v,k in out],values=[val for val,v,k in out])
log(f'Mixed six complete, {len(out)} distinct candidates validated.')
