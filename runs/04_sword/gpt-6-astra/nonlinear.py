import numpy as np
from scipy.stats import chi2
from scipy.spatial.distance import pdist
X=np.load('analysis_outputs/X.npy'); n,d=X.shape
f=open('analysis_outputs/log.txt','a',buffering=1)
def log(s):print(s,flush=True);f.write(str(s)+'\n')
# Full binned pair contingency dependence and correlations of transformed features
out=[]
for b in [4,8,16]:
 Q=np.minimum((X*b).astype(int),b-1)
 for i in range(d):
  for j in range(i):
   H=np.bincount(Q[:,i]*b+Q[:,j],minlength=b*b).reshape(b,b)
   E=H.sum(0)[None,:]*H.sum(1)[:,None]/n
   stat=np.sum((H-E)**2/E);p=chi2.sf(stat,(b-1)**2)
   out.append((p,b,i+1,j+1,stat))
log('Top pairwise contingency p-values '+str(sorted(out)[:12]))
# Check triple sign/bit parity, and continuous products
for b in [1,2,4,8,16,32]:
 S=(2*((X*b%1)>.5)-1).astype(float)
 R=[]
 for i in range(d):
  for j in range(i):
   a=(S[:,i]*S[:,j])@S/n
   for k in range(j):R.append((abs(a[k]),i+1,j+1,k+1,a[k]))
 log('bit triple scale '+str(b)+': '+str(sorted(R,reverse=True)[:3]))
