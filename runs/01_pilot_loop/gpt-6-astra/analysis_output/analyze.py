import numpy as np, pandas as pd, scipy.stats as st, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
X=pd.read_csv('X.csv').values
np.save('analysis_output/X.npy',X)
with open('analysis_output/work_log.txt','w') as f:
 f.write('Analysis restricted to X.csv. No other input files inspected.\n')
 f.write(f'Shape {X.shape}; missing {np.isnan(X).sum()}\n')
 f.write('Column summaries (min,max,mean,std,skew,kurtosis):\n')
 for j in range(X.shape[1]):
  f.write(f'x{j+1}: '+str((X[:,j].min(),X[:,j].max(),X[:,j].mean(),X[:,j].std(),st.skew(X[:,j]),st.kurtosis(X[:,j])))+'\n')
 C=np.corrcoef(X.T); np.fill_diagonal(C,0)
 f.write(f'Largest absolute Pearson correlation {np.max(abs(C))}\n')
 f.write('Correlation eigenvalues '+str(np.linalg.eigvalsh(np.corrcoef(X.T)))+'\n')
print(Path('analysis_output/work_log.txt').read_text())
fig,axs=plt.subplots(5,6,figsize=(16,12))
for j,ax in enumerate(axs.flat):ax.hist(X[:,j],50);ax.set_title(f'x{j+1}')
fig.tight_layout();fig.savefig('analysis_output/marginals.png');plt.close(fig)
fig,ax=plt.subplots(figsize=(8,7));im=ax.imshow(C,vmin=-.1,vmax=.1,cmap='coolwarm');fig.colorbar(im);fig.savefig('analysis_output/correlations.png')
