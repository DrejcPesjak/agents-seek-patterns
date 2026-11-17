import numpy as np, scipy.stats as st, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import time,json
X=np.loadtxt('X.csv',delimiter=',',skiprows=1)
def log(s):
 print(s,flush=True)
 with open('analysis.log','a') as f:f.write(s+'\n')
log('Read only X.csv. Shape '+str(X.shape)+'. All analyses use this data or independent simulated noise.')
log('Ranges '+str(np.round(np.c_[X.min(0),X.max(0)],4)))
log('Means '+str(np.round(X.mean(0),4)))
log('Std '+str(np.round(X.std(0),4)))
C=np.corrcoef(X.T);np.fill_diagonal(C,0)
log('Max abs Pearson correlation '+str(np.abs(C).max()))
log('Covariance eigenvalues '+str(np.linalg.eigvalsh(np.cov(X.T))))
fig,ax=plt.subplots(5,6,figsize=(16,12))
for j,a in enumerate(ax.flat):a.hist(X[:,j],bins=30);a.set_title('x'+str(j+1))
fig.tight_layout();fig.savefig('01_marginals.png');plt.close(fig)
# All pairs binned mutual dependence, including diagonal/curved patterns.
scores=[]
for i in range(30):
 for j in range(i):
  h=np.histogram2d(X[:,i],X[:,j],bins=16,range=[[0,1],[0,1]])[0]
  ex=h.sum(0)[None,:]*h.sum(1)[:,None]/len(X)
  chi=((h-ex)**2/(ex+1e-12)).sum()
  scores.append((chi,i,j))
scores.sort(reverse=True);log('Top pair chi-square (225 df): '+str(scores[:15]))
fig,ax=plt.subplots(2,4,figsize=(16,8))
for a,(s,i,j) in zip(ax.flat,scores):a.scatter(X[:,j],X[:,i],s=1);a.set(xlabel=f'x{j+1}',ylabel=f'x{i+1}',title=f'chi2={s:.1f}')
fig.tight_layout();fig.savefig('02_pair_candidates.png');plt.close(fig)
np.savez('working_data.npz',X=X)
