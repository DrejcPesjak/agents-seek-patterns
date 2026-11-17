import numpy as np,warnings
from scipy import stats,special
from sklearn.decomposition import FastICA,PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
X=np.load('analysis_outputs/X.npy');rng=np.random.default_rng(4363)
f=open('analysis_outputs/log.txt','a',buffering=1)
def log(s):print(s,flush=True);f.write(str(s)+'\n')
# Projection pursuit, using Gaussian marginal scores to remove the original uniform marginals.
Y=special.ndtri(X)
for fun in ['logcosh','exp','cube']:
 with warnings.catch_warnings(record=True) as ws:
  model=FastICA(n_components=30,fun=fun,max_iter=800,tol=.001,random_state=4,whiten='unit-variance')
  try:model.fit(Y[:3000])
  except ValueError:
   model=FastICA(n_components=30,fun=fun,max_iter=800,tol=.001,random_state=4,whiten=True);model.fit(Y[:3000])
 tr=model.transform(Y[:3000]);te=model.transform(Y[5000:]);p=[stats.normaltest(te[:,j]).pvalue for j in range(30)]
 log(f'Gaussian-score ICA {fun}: smallest held-out normality p={min(p):.6g} among 30 components; train max excess kurtosis={max(abs(stats.kurtosis(tr))):.5f}; convergence warnings={len(ws)}')
# Visual overview and reproducible noise comparison
fig,axs=plt.subplots(2,3,figsize=(14,8))
C=np.corrcoef(X.T);im=axs[0,0].imshow(C-np.eye(30),vmin=-.05,vmax=.05,cmap='coolwarm');fig.colorbar(im,ax=axs[0,0]);axs[0,0].set_title('Off-diagonal Pearson correlations')
e=np.linalg.eigvalsh(np.cov(X.T))[::-1];U=rng.random(X.shape);e0=np.linalg.eigvalsh(np.cov(U.T))[::-1]
axs[0,1].plot(e,'o-',label='X.csv');axs[0,1].plot(e0,'o-',label='independent uniform');axs[0,1].legend();axs[0,1].set_title('Covariance eigenvalues')
p=PCA(2).fit_transform(X);axs[0,2].scatter(p[:,0],p[:,1],s=2,alpha=.25);axs[0,2].set_title('First two principal components')
axs[1,0].scatter(X[:,0],X[:,1],s=2,alpha=.25);axs[1,0].set(xlabel='x1',ylabel='x2',title='Example raw projection')
axs[1,1].scatter((X[:,0]+X[:,1])%1,(X[:,2]+X[:,3])%1,s=2,alpha=.25);axs[1,1].set_title('Example modulo-sum projection')
axs[1,2].hist([stats.kstest(v,'uniform').pvalue for v in X.T]);axs[1,2].set_title('Marginal uniformity p-values');axs[1,2].set_xlim(0,1)
fig.tight_layout();fig.savefig('analysis_outputs/overview.png',dpi=150)
