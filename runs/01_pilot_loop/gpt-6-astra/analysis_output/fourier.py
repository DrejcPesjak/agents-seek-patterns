import os
os.environ['MPLCONFIGDIR']='/tmp/mpl-cache'
import numpy as np,pandas as pd,time
from itertools import combinations
X=pd.read_csv('X.csv',header=None).values
np.save('analysis_output/X.npy',X)
n,d=X.shape
with open('analysis_output/work_log.txt','a') as f:f.write(f'Correction: file has no header; true shape {X.shape}. User authorized all 50 columns.\n')
Z=np.exp(2j*np.pi*X).astype('complex64')
# Features: positive first coefficient, +/- second coefficient, harmonics 1..3.
F=[]; desc=[]
for i,j in combinations(range(d),2):
 for a,b in [(1,1),(1,-1),(1,2),(1,-2),(2,1),(2,-1),(2,2),(2,-2),(1,3),(1,-3),(3,1),(3,-1)]:
  F.append(Z[:,i]**a*Z[:,j]**b);desc.append([(i+1,a),(j+1,b)])
F=np.array(F).T.copy();desc=np.array(desc)
one=[]
for k in range(1,5):
 for j in range(d):one.append((abs(np.mean(Z[:,j]**k)),[(j+1,k)]))
pairs=sorted([(abs(v),de.tolist()) for v,de in zip(F.mean(axis=0),desc)],reverse=True)
print('single',sorted(one,reverse=True)[:10]); print('pairs',pairs[:20],flush=True)
# triples: pair with each variable at harmonic 1..3
out=[]
for k in range(1,4):
 for sign in [-1,1]:
  C=F.T@(Z**(k*sign))/n
  for idx,j in zip(*np.where(abs(C)>.065)):
   de=desc[idx].tolist()
   if j+1 not in [t[0] for t in de]:out.append((float(abs(C[idx,j])),de+[[j+1,k*sign]]))
out.sort(reverse=True);print('triples',out[:50],flush=True)
np.savez('analysis_output/fourier_search.npz',pairs=np.array(pairs,dtype=object),triples=np.array(out,dtype=object))
# quadruple with unit coefficients, blocked pair-pair products
ids=np.where(np.all(abs(desc[:,:,1])==1,axis=1))[0];G=F[:,ids];de=desc[ids]
out4=[]
for start in range(0,len(ids),128):
 for sign in [-1,1]:
  C=G[:,start:start+128].T@(G if sign==1 else G.conj())/n
  aa,bb=np.where(abs(C)>.075)
  for a,b in zip(aa,bb):
   a+=start
   if len(set(de[a,:,0])|set(de[b,:,0]))<4:continue
   out4.append((float(abs(C[a-start,b])),de[a].tolist()+[[int(v),int(c*sign)] for v,c in de[b]]))
 print('quad block',start,'found',len(out4),flush=True)
out4.sort(reverse=True)
print('quadruples',out4[:80],flush=True)
np.save('analysis_output/fourier_quad.npy',np.array(out4,dtype=object))
