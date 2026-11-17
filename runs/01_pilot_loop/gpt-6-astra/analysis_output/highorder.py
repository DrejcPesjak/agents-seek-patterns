import numpy as np,time
from itertools import combinations
X=np.load('analysis_output/X.npy');n,d=X.shape
rng=np.random.default_rng(174);perm=rng.permutation(n)
Z=np.exp(2j*np.pi*X).astype('complex64')
desc=[]
for t in combinations(range(d),3):
 for s1 in [-1,1]:
  for s2 in [-1,1]:desc.append([(t[0],1),(t[1],s1),(t[2],s2)])
desc=np.array(desc);M=len(desc);nd=384
A=np.ones((nd,M),dtype='complex64')
for k in range(3):A*=Z[perm[:nd]][:,desc[:,k,0]]**desc[:,k,1]
pdesc=[]
for i,j in combinations(range(d),2):
 for s in [-1,1]:pdesc.append([(i,1),(j,s)])
pdesc=np.array(pdesc);P=Z[perm[:nd]][:,pdesc[:,0,0]]*Z[perm[:nd]][:,pdesc[:,1,0]]**pdesc[:,1,1]
results=[]
def check(aa,bb,C,DD,offset,sign,kind):
 for a,b in zip(aa,bb):
  a+=offset
  de=np.concatenate([desc[a],DD[b]*np.array([1,sign])])
  if len(set(de[:,0]))<len(de):continue
  v=np.ones(n-nd,dtype='complex64')
  for j,c in de:v*=Z[perm[nd:],j]**c
  sc=abs(v.mean())
  if sc>.065:
   results.append((float(sc),float(abs(C[a-offset,b])),(de+np.array([1,0])).tolist()))
   print('FOUND',kind,results[-1],flush=True)
t=time.time()
for start in range(0,M,256):
 for sign in [-1,1]:
  C=A[:,start:start+256].T@(P if sign==1 else P.conj())/nd
  aa,bb=np.where(abs(C)>.215);check(aa,bb,C,pdesc,start,sign,5)
 if start%8192==0:print('five',start,time.time()-t,flush=True)
print('five complete',results,flush=True)
# Six-column unit frequencies. Discovery scan, independent validation.
for start in range(0,M,256):
 for sign in [-1,1]:
  C=A[:,start:start+256].T@(A[:,start:] if sign==1 else A[:,start:].conj())/nd
  aa,bb=np.where(abs(C)>.235)
  check(aa,bb,C,desc[start:],start,sign,6)
 if start%4096==0:print('six',start,time.time()-t,flush=True)
results.sort(reverse=True);print('RESULT',results[:100],flush=True)
np.save('analysis_output/highorder.npy',np.array(results,dtype=object))
