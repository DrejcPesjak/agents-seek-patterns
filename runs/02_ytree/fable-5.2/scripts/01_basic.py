import numpy as np, pandas as pd
from scipy import stats
X = pd.read_csv('X.csv').values
n,d = X.shape
print(X.shape, X.min(), X.max())
print('mean', X.mean(0).round(3)); print('std', X.std(0).round(4), 'unif std', 1/12**.5)
# KS vs uniform
print('KS p', np.array([stats.kstest(X[:,j],'uniform').pvalue for j in range(d)]).round(3))
C = np.corrcoef(X.T); np.fill_diagonal(C,0)
print('max abs corr', np.abs(C).max(), np.unravel_index(np.abs(C).argmax(), C.shape), 'expected max ~', 3.5/np.sqrt(n))
ev = np.linalg.eigvalsh(np.cov(X.T)); print('PCA eig', ev.round(4))
# decimals/duplicates
print('unique per col', [len(np.unique(X[:,j])) for j in range(d)][:10])
# circular tests: mean of exp(2pi i k x)
for k in (1,2,3):
    print('k',k, np.abs(np.exp(2j*np.pi*k*X).mean(0)).round(3).max(), 'expected ~', 1/np.sqrt(n))
# pairwise sum/diff circular dependence
best=[]
for i in range(d):
    for j in range(i+1,d):
        for a,b in [(1,1),(1,-1),(1,2),(2,1),(1,-2),(2,-1)]:
            r=abs(np.exp(2j*np.pi*(a*X[:,i]+b*X[:,j])).mean())
            best.append((r,i+1,j+1,a,b))
best.sort(reverse=True); print(best[:8])
# rank corr of squares / abs deviations
A=np.abs(X-0.5); C2=np.corrcoef(A.T); np.fill_diagonal(C2,0); print('max abs-dev corr', np.abs(C2).max(), np.unravel_index(np.abs(C2).argmax(), C2.shape))
# serial dependence between rows
print('row autocorr lag1', np.array([np.corrcoef(X[:-1,j],X[1:,j])[0,1] for j in range(d)]).round(3))
