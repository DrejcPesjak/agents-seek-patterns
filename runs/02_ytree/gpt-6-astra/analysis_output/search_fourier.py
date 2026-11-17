import os,time
os.environ['OPENBLAS_NUM_THREADS']='6'
import numpy as np,pandas as pd
from pathlib import Path
X=pd.read_csv('X.csv').to_numpy();N=1400;K=3
log=Path('analysis_output/log.txt')
def out(s):
 print(s,flush=True)
 with log.open('a') as f:f.write(str(s)+'\n')
out(f'Sparse Fourier search: up to four columns, coefficients ±1..{K}, discovery subsample {N}; rank on all 3500 discovery rows and independently check 1500 validation rows.')
meta=[];cols=[]
for a in range(30):
 for b in range(a+1,30):
  for k in range(1,K+1):
   for l in [v for v in range(-K,K+1) if v]:
    meta.append((a,k,b,l));cols.append(np.exp(2j*np.pi*(k*X[:N,a]+l*X[:N,b])))
Z=np.array(cols,dtype=np.complex64).T.copy();del cols
meta=np.array(meta); M=Z.shape[1];out(f'{M} pair characters')
candidates={}
def add(v,score):
 v=np.asarray(v,dtype=np.int16)
 if not np.any(v):return
 if v[np.flatnonzero(v)[0]]<0:v=-v
 key=tuple(v)
 if key not in candidates or score>candidates[key]:candidates[key]=float(score)
# triples, arbitrary coeff up to 8 for single
sm=[(a,k) for a in range(30) for k in range(1,9)]
S=np.array([np.exp(2j*np.pi*k*X[:N,a]) for a,k in sm],dtype=np.complex64).T
for sign in [-1,1]:
 R=(Z.T.conj() if sign==-1 else Z.T)@S/N
 inds=np.argpartition(abs(R).ravel(),-3000)[-3000:]
 for ind in inds:
  i,j=divmod(ind,len(sm));a,k,b,l=meta[i];c,m=sm[j]
  if c in [a,b]:continue
  v=np.zeros(30,dtype=int);v[a]=sign*k;v[b]=sign*l;v[c]=m
  add(v,abs(R[i,j]))
out('Triple stage done')
t=time.time()
for start in range(0,M,160):
 for sign in [-1,1]:
  R=(Z[:,start:start+160].T.conj() if sign==-1 else Z[:,start:start+160].T)@Z/N
  A=abs(R)
  # filter overlapping columns and duplicate ordering before selection
  for ii in range(R.shape[0]):
   i=start+ii;a,k,b,l=meta[i]
   invalid=(meta[:,0]==a)|(meta[:,0]==b)|(meta[:,2]==a)|(meta[:,2]==b)|(np.arange(M)<=i)
   A[ii,invalid]=0
  inds=np.argpartition(A.ravel(),-100)[-100:]
  for ind in inds:
   ii,j=divmod(ind,M);i=start+ii;a,k,b,l=meta[i];c,m,d,n=meta[j]
   v=np.zeros(30,dtype=int);v[a]=sign*k;v[b]=sign*l;v[c]=m;v[d]=n
   add(v,A[ii,j])
 if (start//160)%10==0:out(f'block {start}/{M}, {time.time()-t:.1f}s, {len(candidates)} candidates')
V=np.array(list(candidates)); screen=np.array(list(candidates.values()))
np.savez('analysis_output/fourier4_candidates.npz',V=V,screen=screen)
train=[];test=[]
for start in range(0,len(V),500):
 train.extend(abs(np.exp(2j*np.pi*(X[:3500]@V[start:start+500].T)).mean(0)))
 test.extend(abs(np.exp(2j*np.pi*(X[3500:]@V[start:start+500].T)).mean(0)))
train=np.array(train);test=np.array(test);np.savez('analysis_output/fourier4_results.npz',V=V,screen=screen,train=train,test=test)
for i in np.argsort(-train)[:50]:out(f'{train[i]:.5f} validation {test[i]:.5f} screen {screen[i]:.5f} '+str([(j+1,int(k)) for j,k in enumerate(V[i]) if k]))
