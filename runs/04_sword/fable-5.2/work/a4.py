import numpy as np, pandas as pd, itertools
from scipy import stats
from scipy.spatial import cKDTree
X=pd.read_csv('../X.csv').values; n,d=X.shape
rng=np.random.default_rng(0); U=rng.random((n,d))
def feats(A):
    S=np.sort(A,1)
    return dict(sum=A.sum(1),sumfrac=A.sum(1)%1,ss=(A**2).sum(1),mx=S[:,-1],mn=S[:,0],med=S[:,15],mingap=np.diff(S,axis=1).min(1),maxgap=np.diff(S,axis=1).max(1),
      altsum=(A*(-1)**np.arange(d)).sum(1)%1, prod=np.log(A+1e-300).sum(1), lag1=(A[:,1:]*A[:,:-1]).sum(1), nlow=(A<0.1).sum(1), nhigh=(A>0.9).sum(1))
fx,fu=feats(X),feats(U)
for k in fx: print(k,'KS p=%.3g'%stats.ks_2samp(fx[k],fu[k]).pvalue)
# full NN (torus & plain)
for name,A in [('X',X),('U',U)]:
    dd,_=cKDTree(A).query(A,k=2); dt,_=cKDTree(A,boxsize=1).query(A,k=2)
    print(name,'NN plain',np.percentile(dd[:,1],[0,1,50,99]),'torus',np.percentile(dt[:,1],[0,1,50,99]))
# pair NN on torus
r=[]
for i,j in itertools.combinations(range(d),2):
    dd,_=cKDTree(X[:,[i,j]],boxsize=1).query(X[:,[i,j]],k=2); r.append((dd[:,1].mean(),i+1,j+1))
r.sort(); m=np.array([a for a,_,_ in r]); print('pairNN lowest',r[:5],'mean',m.mean(),'sd',m.std())
r=[]
for c in itertools.combinations(range(d),3):
    dd,_=cKDTree(X[:,c],boxsize=1).query(X[:,c],k=2); r.append((dd[:,1].mean(),)+c)
r.sort(); m=np.array([a[0] for a in r]); print('tripNN lowest',r[:5],'mean',m.mean(),'sd',m.std(),'z',(m[0]-m.mean())/m.std())
# row order
for j in range(3): print('autocorr rows',np.corrcoef(X[1:,j],X[:-1,j])[0,1])
