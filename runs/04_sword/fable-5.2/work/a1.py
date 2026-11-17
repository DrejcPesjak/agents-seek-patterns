import numpy as np, pandas as pd
from scipy import stats
X=pd.read_csv('../X.csv').values; print(X.shape)
np.set_printoptions(precision=3,suppress=True,linewidth=200)
print('mean',X.mean(0));print('std',X.std(0));print('min',X.min(0));print('max',X.max(0))
print('skew',stats.skew(X));print('kurt',stats.kurtosis(X))
print('nuniq',[len(np.unique(c)) for c in X.T])
C=np.corrcoef(X.T);np.fill_diagonal(C,0);print('maxcorr',np.abs(C).max(), np.unravel_index(np.abs(C).argmax(),C.shape))
ev=np.linalg.eigvalsh(np.cov(X.T));print('eig',ev)
# fractional part / decimals
print('frac sample',X[:3,:6])
for j in range(30):
    print(j+1,'KS norm p=%.3g'%stats.kstest((X[:,j]-X[:,j].mean())/X[:,j].std(),'norm').pvalue,'unif p=%.3g'%stats.kstest(X[:,j],'uniform',args=(X[:,j].min(),np.ptp(X[:,j]))).pvalue)
