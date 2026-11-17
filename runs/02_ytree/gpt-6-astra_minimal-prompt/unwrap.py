import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
X=np.load('working_data.npz')['X'];V=np.load('fourier_results.npz')['results'][:2];Y=X@V.T%1
print('Quantiles',np.quantile(Y,[0,.001,.01,.1,.5,.9,.99,.999,1],axis=0))
fig,ax=plt.subplots(figsize=(9,9));ax.scatter(*Y.T,s=3,alpha=.5);ax.set_aspect('equal');ax.set(xlabel='(2*x10+2*x23+x25+x30)%1',ylabel='(2*x1-2*x6-x11-x24)%1');ax.grid(alpha=.2);fig.tight_layout();fig.savefig('04_unwrapped.png');np.save('coordinates.npy',Y)
