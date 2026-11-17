import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
X=pd.read_csv('X.csv')
a=X.to_numpy(float); n,d=a.shape
Path('plots').mkdir(exist_ok=True)
def log(s):
 print(s,flush=True)
 with open('analysis.log','a') as f:f.write(str(s)+'\n')
log('Only input: X.csv. All other files used in analysis are generated outputs. Seed=321.')
log(f'Shape {a.shape}; columns {list(X.columns)}; missing {np.isnan(a).sum()}')
log(X.describe(percentiles=[.01,.25,.5,.75,.99]).to_string())
c=np.corrcoef(a.T); log('Correlation largest off diagonal '+str(np.max(np.abs(c-np.eye(d)))))
z=(a-a.mean(0))/a.std(0)
s=np.linalg.svd(z,compute_uv=False);log('PCA eigenvalues '+str(s*s/n))
fig,axs=plt.subplots(5,6,figsize=(18,12))
for j,ax in enumerate(axs.flat):
 if j<d: ax.hist(a[:,j],bins=50);ax.set_title(X.columns[j])
fig.tight_layout();fig.savefig('plots/marginals.png');plt.close(fig)
fig,ax=plt.subplots(figsize=(8,7));im=ax.imshow(c,vmin=-1,vmax=1,cmap='coolwarm');fig.colorbar(im);fig.savefig('plots/correlations.png');plt.close(fig)
np.save('data.npy',a)
