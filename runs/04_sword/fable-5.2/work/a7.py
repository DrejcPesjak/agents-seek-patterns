import numpy as np, pandas as pd
from scipy import stats
from sklearn.decomposition import FastICA
X=pd.read_csv('../X.csv').values; n,d=X.shape
rng=np.random.default_rng(1); U=rng.random((n,d))
for name,A in [('X',X),('U',U)]:
    Z=stats.norm.ppf(np.clip(A,1e-12,1-1e-12))
    print(name,'cov eig',np.round(np.linalg.eigvalsh(np.cov(Z.T))[[0,1,2,-3,-2,-1]],3))
    for fun in ['logcosh','cube']:
        S=FastICA(n_components=d,fun=fun,random_state=0,max_iter=2000,tol=1e-6).fit_transform(Z)
        k=stats.kurtosis(S); sk=stats.skew(S)
        print(name,fun,'kurt sorted',np.round(np.sort(k)[[0,1,2,-3,-2,-1]],3),'skew',np.round(np.sort(sk)[[0,1,-2,-1]],3))
    # 4th-moment tensor test: for centered X too
    C=A-0.5
    S=FastICA(n_components=d,fun='cube',random_state=0,max_iter=2000).fit_transform(C)
    print(name,'raw ICA kurt',np.round(np.sort(stats.kurtosis(S))[[0,1,2,-3,-2,-1]],3))
