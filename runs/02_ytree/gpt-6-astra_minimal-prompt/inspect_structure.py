import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
X=np.load('working_data.npz')['X'];V=np.load('fourier_results.npz')['results'][:2]
Y=(X@V.T+.5)%1-.5
print('Vectors',V);print('Quantiles',np.quantile(Y,[0,.001,.01,.1,.5,.9,.99,.999,1],axis=0));print('Mean and covariance',Y.mean(0),np.cov(Y.T))
fig,ax=plt.subplots(1,3,figsize=(17,5));ax[0].scatter(*Y.T,s=2,alpha=.5);ax[0].set_aspect('equal');ax[0].set(xlabel='2*x10+2*x23+x25+x30 (centered mod 1)',ylabel='2*x1-2*x6-x11-x24 (centered mod 1)')
for j in range(2):ax[j+1].hist(Y[:,j],bins=80)
fig.tight_layout();fig.savefig('03_recovered_coordinates.png');np.save('coordinates.npy',Y)
