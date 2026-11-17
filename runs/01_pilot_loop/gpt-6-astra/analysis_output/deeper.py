import numpy as np,time
from itertools import combinations
from scipy.stats import chi2
X=np.load('analysis_output/X.npy');n,d=X.shape
Z=np.exp(2j*np.pi*X).astype('complex64')
desc=[];F=[]
for i,j in combinations(range(d),2):
 for a in range(1,4):
  for b in [-3,-2,-1,1,2,3]:
   F.append(Z[:,i]**a*Z[:,j]**b);desc.append([(i+1,a),(j+1,b)])
F=np.array(F).T.copy();desc=np.array(desc)
# Quad search on 768 discovery rows, validate promising coefficients on independent 3232 rows.
# If true amplitude >= .2, discovery screen .14 has high power.
rng=np.random.default_rng(174);perm=rng.permutation(n);A=F[perm[:768]];B=F[perm[768:]]
out=[];t=time.time()
for start in range(0,F.shape[1],256):
 for sign in [-1,1]:
  C=A[:,start:start+256].T@(A if sign==1 else A.conj())/len(A)
  aa,bb=np.where(abs(C)>.14)
  for a,b in zip(aa,bb):
   a+=start
   if len(set(desc[a,:,0])|set(desc[b,:,0]))<4:continue
   v=np.mean(B[:,a]*(B[:,b] if sign==1 else B[:,b].conj()))
   if abs(v)>.055:
    de=desc[a].tolist()+[[int(i),int(c*sign)] for i,c in desc[b]]
    out.append((float(abs(v)),float(abs(C[a-start,b])),de));print('candidate',out[-1],flush=True)
 if start%2048==0:print('progress',start,'elapsed',time.time()-t,flush=True)
out.sort(reverse=True);print('RESULT',out[:100],flush=True)
np.save('analysis_output/weighted_quad.npy',np.array(out,dtype=object))
# 2D histogram independence; Pearson test calibrated to permutation
mi=[]
for i,j in combinations(range(d),2):
 h=np.histogram2d(X[:,i],X[:,j],bins=12,range=[[0,1],[0,1]])[0]
 e=h.sum(1)[:,None]*h.sum(0)[None,:]/n
 s=np.sum((h-e)**2/e)
 mi.append((float(chi2.sf(s,121)),i+1,j+1,float(s)))
print('PAIR CHI2',sorted(mi)[:30],flush=True)
