import os
os.environ['OPENBLAS_NUM_THREADS']='4'
import numpy as np,pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
X=pd.read_csv('X.csv').to_numpy(); n,d=X.shape
log=Path('analysis_output/log.txt')
def out(s):
 print(s,flush=True)
 with log.open('a') as f:f.write(str(s)+'\n')
out('Initial analysis: read only X.csv; all generated work goes into analysis_output. Split rows 0:3500 discovery, 3500:5000 validation.')
out(f'Shape {X.shape}, missing {np.isnan(X).sum()}')
out('Column min/max/mean/std/uniform KS p:')
for j in range(d):out(f'x{j+1}: {X[:,j].min():.5f} {X[:,j].max():.5f} {X[:,j].mean():.5f} {X[:,j].std():.5f} {stats.kstest(X[:,j],"uniform").pvalue:.4g}')
C=np.corrcoef(X.T);np.fill_diagonal(C,0)
out(f'Max absolute Pearson correlation {np.abs(C).max():.6f} at {np.unravel_index(abs(C).argmax(),C.shape)}')
s=np.linalg.eigvalsh(np.cov(X.T));out(f'Covariance eigenvalues {s}')
fig,ax=plt.subplots(1,3,figsize=(15,4));ax[0].hist(X.ravel(),bins=70);ax[0].set_title('Pooled marginal values');im=ax[1].imshow(C,vmin=-.1,vmax=.1,cmap='coolwarm');fig.colorbar(im,ax=ax[1]);ax[1].set_title('Pearson correlations');ax[2].plot(s[::-1],'o-');ax[2].set_title('Covariance eigenvalues');fig.tight_layout();fig.savefig('analysis_output/01_initial.png',dpi=150)
# integer Fourier characters with at most two nonzero coordinates; use discovery and validation separately
K=8; meta=[(j,k) for j in range(d) for k in range(1,K+1)]
Z=np.exp(2j*np.pi*X[:3500,:,None]*np.arange(1,K+1)).reshape(3500,-1)
R=Z.T.conj()@Z/3500; S=Z.T@Z/3500
records=[]
for a,(j,k) in enumerate(meta):
 records.append((abs(Z[:,a].mean()),[(j,k)]))
 for b in range(a+1,len(meta)):
  l,m=meta[b]
  if j==l:continue
  records.append((abs(R[a,b]),[(j,-k),(l,m)]))
  records.append((abs(S[a,b]),[(j,k),(l,m)]))
records.sort(key=lambda t:-t[0])
out('Top Fourier modes (discovery magnitude, validation magnitude, integer vector):')
for strength,v in records[:40]:
 val=abs(np.exp(2j*np.pi*sum(X[3500:,j]*k for j,k in v)).mean())
 out(f'{strength:.6f} {val:.6f} '+str([(j+1,k) for j,k in v]))
np.savez('analysis_output/fourier_pairs.npz',strength=np.array([r[0] for r in records]),vectors=np.array([str(r[1]) for r in records]))
# 2D histogram chi-square statistics
rec=[]
for j in range(d):
 for k in range(j):
  h=np.histogram2d(X[:,j],X[:,k],bins=12,range=[[0,1],[0,1]])[0]
  expected=h.sum(0)[None,:]*h.sum(1)[:,None]/n
  chi=((h-expected)**2/expected).sum()
  rec.append((chi,j+1,k+1))
out('Largest pairwise binned dependence scores: '+str(sorted(rec,reverse=True)[:15]))
