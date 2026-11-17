import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time,json
X=np.genfromtxt('X.csv',delimiter=',',skip_header=1)
np.save('analysis_outputs/X.npy',X)
f=open('analysis_outputs/log.txt','a',buffering=1)
def log(s): print(s,flush=True);f.write(str(s)+'\n')
log('Only source read: X.csv. '+str(X.shape))
log('range '+str((X.min(),X.max()))+'; nonfinite '+str(np.sum(~np.isfinite(X))))
log('means '+str(X.mean(0)));log('stds '+str(X.std(0)))
log('uniform KS '+str([round(stats.kstest(v,'uniform').statistic,4) for v in X.T]))
C=np.corrcoef(X.T);np.fill_diagonal(C,0);log('max correlation '+str(np.max(abs(C))))
log('cov eigenvalues '+str(np.linalg.eigvalsh(np.cov(X.T))))
fig,axs=plt.subplots(5,6,figsize=(16,12))
for i,ax in enumerate(axs.flat): ax.hist(X[:,i],bins=40);ax.set_title('x'+str(i+1))
fig.tight_layout();fig.savefig('analysis_outputs/marginals.png');plt.close(fig)
# Fourier pair scans
Z=np.exp(2j*np.pi*X)
for a in range(1,6):
 for b in range(1,6):
  for sign in [1,-1]:
   F=(Z**a).T @ (Z**(b*sign))/len(X)
   np.fill_diagonal(F,0)
   k=np.unravel_index(np.argmax(abs(F)),F.shape)
   if abs(F[k])>.045: log(f'pair Fourier {a} x{k[0]+1} + {b*sign} x{k[1]+1}: {abs(F[k]):.5f}')
log('Initial scans complete')
