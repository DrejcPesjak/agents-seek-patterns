import numpy as np,time
from lll_impl import lll
X=np.load('analysis_outputs/X.npy');f=open('analysis_outputs/log.txt','a',buffering=1)
def log(s):print(s,flush=True);f.write(str(s)+'\n')
log('Lattice reduction tests for dense, small-integer modular linear constraints. Candidate selection uses only first 41 rows; validation uses rows 2000:5000.')
for m,scale in [(8,10**8),(16,10**6),(24,1000),(40,100)]:
 B=np.zeros((30+m,30+m),dtype=np.int64);B[:30,:30]=np.eye(30,dtype=int);B[:30,30:]=np.rint((X[1:m+1]-X[0]).T*scale).astype(np.int64);B[30:,30:]=scale*np.eye(m,dtype=np.int64)
 t=time.time();R=np.array(lll(B)).astype(np.int64);out=[]
 for r in R:
  k=r[:30]
  if np.any(k):
   val=abs(np.exp(2j*np.pi*(X[2000:5000]@k)).mean());out.append((val,k.tolist()))
 out.sort(reverse=True);log(f'Lattice m={m}, scale={scale}, seconds={time.time()-t:.1f}, strongest validation '+str(out[:2]))
