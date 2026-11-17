import numpy as np, pandas as pd
from scipy import stats
X=pd.read_csv('X.csv').values
print(X.shape, np.isnan(X).sum())
print('min',X.min(0).round(3)); print('max',X.max(0).round(3))
print('mean',X.mean(0).round(3)); print('std',X.std(0).round(3))
print('KS vs U(0,1) p:',np.array([stats.kstest(X[:,j],'uniform').pvalue for j in range(30)]).round(3))
C=np.corrcoef(X.T); np.fill_diagonal(C,0)
print('max |corr|',np.abs(C).max(), np.unravel_index(np.abs(C).argmax(),C.shape))
ev=np.linalg.eigvalsh(np.cov(X.T)); print('cov eig',ev.round(4))
# duplicates / precision
print('unique per col',[len(np.unique(X[:,j])) for j in range(30)])
# row order autocorr
print('lag1 autocorr',np.array([np.corrcoef(X[:-1,j],X[1:,j])[0,1] for j in range(30)]).round(3))
