import numpy as np
X=np.loadtxt('example_X.csv',delimiter=',')
print(X.shape, X.min(), X.max(), X.mean(), X.std())
print('col mean range',X.mean(0).min(),X.mean(0).max(),'col std range',X.std(0).min(),X.std(0).max())
print('unique vals per col (first 10)',[len(np.unique(X[:,j])) for j in range(min(10,X.shape[1]))])
Xc=X-X.mean(0)
s=np.linalg.svd(Xc,compute_uv=False)
print('top sv',np.round(s[:12],2)); print('bottom sv',np.round(s[-5:],2))
C=np.corrcoef(X.T); np.fill_diagonal(C,0)
print('max abs corr',np.abs(C).max(), 'expected noise ~',4/np.sqrt(X.shape[0]))
# uniformity
from scipy import stats
print('KS vs uniform per col, min p',min(stats.kstest(X[:,j],'uniform').pvalue for j in range(X.shape[1])))
