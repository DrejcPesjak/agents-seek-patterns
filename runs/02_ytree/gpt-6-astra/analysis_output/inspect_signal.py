import os
os.environ['OPENBLAS_NUM_THREADS']='4'
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
X=pd.read_csv('X.csv').to_numpy();r=np.load('analysis_output/fourier4_results.npz');ids=np.argsort(-r['train'])[:2];V=r['V'][ids];U=(X@V.T)%1
# circular center and unwrap
center=np.angle(np.exp(2j*np.pi*U).mean(0))/(2*np.pi)%1
Y=(U-center+.5)%1-.5
with Path('analysis_output/log.txt').open('a') as f:
 for msg in [f'Found two strong independent four-column Fourier modes: {V.tolist()}',f'Circular centers {center}; unwrapped quantiles {np.quantile(Y,[0,.01,.5,.99,1],axis=0)}',f'Fourier magnitudes: discovery {r["train"][ids]}, validation {r["test"][ids]}']:
  print(msg);f.write(msg+'\n')
np.savez('analysis_output/signal.npz',V=V,U=U,center=center,Y=Y)
fig,ax=plt.subplots(1,3,figsize=(16,5));ax[0].scatter(U[:,0],U[:,1],s=2,alpha=.5);ax[0].set(xlim=(0,1),ylim=(0,1),xlabel='(2*x10+2*x23+x25+x30) mod 1',ylabel='(2*x1-2*x6-x11-x24) mod 1',title='Discovered modular coordinates');ax[1].scatter(Y[:3500,0],Y[:3500,1],s=2,alpha=.4);ax[1].set(title='Unwrapped discovery rows',xlabel='First coordinate, centered',ylabel='Second coordinate, centered');ax[1].axis('equal');ax[2].scatter(Y[3500:,0],Y[3500:,1],s=3,alpha=.5,color='orange');ax[2].axis('equal');ax[2].set(title='Independent validation rows');fig.tight_layout();fig.savefig('analysis_output/02_discovered_projection.png',dpi=180)
