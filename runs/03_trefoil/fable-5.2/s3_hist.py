import numpy as np, pandas as pd, itertools
from scipy import stats
X=pd.read_csv('X.csv').values; n,d=X.shape
# 2D histograms, several resolutions
for B in (8,20,40):
    I=np.minimum((X*B).astype(int),B-1); out=[]
    for i,j in itertools.combinations(range(d),2):
        c=np.bincount(I[:,i]*B+I[:,j],minlength=B*B); e=n/B/B
        out.append((((c-e)**2/e).sum(),c.max(),i+1,j+1))
    out.sort(reverse=True); df=B*B-1
    print('2D B',B,'df',df,'top z:',[(round((o[0]-df)/np.sqrt(2*df),2),o[2],o[3]) for o in out[:5]],'maxcell',max(o[1] for o in out))
for B in (4,6,10):
    I=np.minimum((X*B).astype(int),B-1); out=[]
    for i,j,k in itertools.combinations(range(d),3):
        c=np.bincount((I[:,i]*B+I[:,j])*B+I[:,k],minlength=B**3); e=n/B**3
        out.append((((c-e)**2/e).sum(),c.max(),i+1,j+1,k+1))
    out.sort(reverse=True); df=B**3-1
    print('3D B',B,'df',df,'top z:',[(round((o[0]-df)/np.sqrt(2*df),2),)+o[2:] for o in out[:5]],'maxcell',max(o[1] for o in out))
# full-dim kNN distances vs uniform simulation
from sklearn.neighbors import NearestNeighbors
def nn(Z,k=5): return NearestNeighbors(n_neighbors=k+1).fit(Z).kneighbors(Z)[0][:,1:]
D=nn(X); R=nn(np.random.default_rng(0).random(X.shape))
print('full kNN d1 quantiles data',np.quantile(D[:,0],[0,.01,.1,.5,.9,1]).round(3))
print('full kNN d1 quantiles unif',np.quantile(R[:,0],[0,.01,.1,.5,.9,1]).round(3))
# single-column fine structure: digits / high-freq Fourier
ks=np.arange(1,2001)
for j in range(d):
    s=n*np.abs(np.exp(2j*np.pi*np.outer(X[:,j],ks)).mean(0))**2
    if s.max()>14: print('col',j+1,'k',ks[s.argmax()],s.max())
print('single-col spectrum done (thr 14, exp max ~',np.log(30*2000),')')
