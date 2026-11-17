import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
X=np.loadtxt('example_X.csv',delimiter=',')
n,d=X.shape; rng=np.random.default_rng(0)
nulls=[rng.random((n,d)) for _ in range(20)]
DC,NC='#1f6feb','#8b949e'
# 1 singular spectrum vs null envelope
sv=lambda A:np.linalg.svd(A-A.mean(0),compute_uv=False)
S=np.array([sv(A) for A in nulls]); s=sv(X)
fig,ax=plt.subplots(figsize=(7,4))
ax.fill_between(range(1,d+1),S.min(0),S.max(0),color=NC,alpha=.4,label='range of 20 uniform-noise matrices')
ax.plot(range(1,d+1),s,'o-',ms=3,color=DC,label='example_X')
ax.set(xlabel='component',ylabel='singular value',title='PCA spectrum: no component stands out of the noise band'); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig('plots/01_pca_spectrum.png',dpi=140); plt.close()
# 2 3D scatter of top-3 PCs
U,_,Vt=np.linalg.svd(X-X.mean(0),full_matrices=False); P=(X-X.mean(0))@Vt[:3].T
fig,axs=plt.subplots(1,3,figsize=(11,3.9))
for ax,(a,b) in zip(axs,[(0,1),(0,2),(1,2)]):
    ax.scatter(P[:,a],P[:,b],s=2,color=DC,alpha=.4); ax.set(xlabel=f'PC{a+1}',ylabel=f'PC{b+1}',aspect='equal')
fig.suptitle('Top-3 PCA projection, three views: a featureless blob')
fig.tight_layout(); fig.savefig('plots/02_pca_top3.png',dpi=140); plt.close()
# 3 pairwise chi2 distribution
def chis(A,nb=8):
    B=np.minimum((A*nb).astype(int),nb-1); o=[]
    for i in range(d):
        for j in range(i+1,d):
            H=np.zeros((nb,nb)); np.add.at(H,(B[:,i],B[:,j]),1); o.append(((H-n/nb**2)**2/(n/nb**2)).sum())
    return np.array(o)
c=chis(X); c0=np.concatenate([chis(A) for A in nulls[:5]])
fig,ax=plt.subplots(figsize=(7,4)); bins=np.linspace(20,130,56)
ax.hist(c0,bins,density=True,color=NC,alpha=.5,label='uniform noise (5 matrices)'); ax.hist(c,bins,density=True,histtype='step',lw=1.8,color=DC,label='example_X (1225 column pairs)')
ax.set(xlabel='chi-square of 8x8 binned column pair',ylabel='density',title='Pairwise dependence: same distribution as noise'); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig('plots/03_pairwise_chi2.png',dpi=140); plt.close()
# 4 most "suspicious" pairs scatter
fig,axs=plt.subplots(1,3,figsize=(11,3.9))
for ax,(i,j) in zip(axs,[(37,42),(6,13),(10,18)]):
    ax.scatter(X[:,i],X[:,j],s=2,color=DC,alpha=.5); ax.set(xlabel=f'col {i}',ylabel=f'col {j}',aspect='equal')
fig.suptitle('The three highest-scoring column pairs still look like uniform squares'); fig.tight_layout(); fig.savefig('plots/04_top_pairs.png',dpi=140); plt.close()
# 5 NN distances
nn=lambda A:NearestNeighbors(n_neighbors=2).fit(A).kneighbors(A)[0][:,1]
fig,ax=plt.subplots(figsize=(7,4)); bins=np.linspace(1.55,2.45,60)
ax.hist(np.concatenate([nn(A) for A in nulls[:5]]),bins,density=True,color=NC,alpha=.5,label='uniform noise'); ax.hist(nn(X),bins,density=True,histtype='step',lw=1.8,color=DC,label='example_X')
ax.set(xlabel='distance to nearest other row',ylabel='density',title='Nearest-neighbour distances: no rows crowd onto a manifold'); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig('plots/05_nn_distances.png',dpi=140); plt.close()
# 6 column-from-others predictability
r=np.load('/tmp/claude-1000/-home-drew99-Projects-agents-and-3d-plots-ex1/4accface-8cd0-43af-9119-b8bc65916e42/scratchpad/r2.npy')
fig,ax=plt.subplots(figsize=(7,3.6)); ax.bar(range(d),r,color=DC); ax.axhline(0,color='k',lw=.8)
ax.set(xlabel='column',ylabel='cross-validated R²',ylim=(-.03,.03),title='Predicting each column from the other 49: R² never above 0')
fig.tight_layout(); fig.savefig('plots/06_column_predictability.png',dpi=140); plt.close()
