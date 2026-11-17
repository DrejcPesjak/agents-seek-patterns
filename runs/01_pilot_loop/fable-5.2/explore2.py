import numpy as np
from scipy import stats
X=np.loadtxt('example_X.csv',delimiter=',')
n,d=X.shape
rng=np.random.default_rng(0)
R=rng.random((n,d))  # null reference
def report(name,f):
    a,b=f(X),f(R); print(f'{name:45s} data={a}  null={b}')
# 1 circular (Fourier) correlations between columns, harmonics 1..3
def circ(X):
    out=[]
    for k in (1,2,3):
        for l in (1,2,3):
            Z=np.exp(2j*np.pi*k*X); W=np.exp(2j*np.pi*l*X)
            M1=np.abs(Z.T@W.conj())/n; M2=np.abs(Z.T@W)/n
            np.fill_diagonal(M1,0) 
            if k==l: np.fill_diagonal(M2,0)
            else: np.fill_diagonal(M2,0)
            out.append(max(M1.max(),M2.max()))
    return np.round(max(out),4)
report('max circular corr (harm<=3)',circ)
# 2 pairwise binned chi2 dependence (8x8)
def chi(X):
    B=np.minimum((X*8).astype(int),7); best=0
    for i in range(d):
        for j in range(i+1,d):
            H=np.zeros((8,8)); np.add.at(H,(B[:,i],B[:,j]),1)
            c=((H-n/64)**2/(n/64)).sum(); best=max(best,c)
    return round(best,1)
report('max pairwise chi2 8x8 (df~49)',chi)
# 3 row-order autocorrelation
def ac(X):
    Xc=X-X.mean(0); return np.round(max(abs((Xc[:-k]*Xc[k:]).mean()/Xc.var()) for k in (1,2,3,5,10)),4)
report('row lag autocorr (all cols pooled)',ac)
def acmax(X):
    Xc=(X-X.mean(0))/X.std(0); return np.round(np.abs((Xc[:-1]*Xc[1:]).mean(0)).max(),4)
report('max per-col lag1 autocorr',acmax)
# 4 nearest neighbour distances (euclid and torus)
from sklearn.neighbors import NearestNeighbors
def nn(X):
    D,_=NearestNeighbors(n_neighbors=3).fit(X).kneighbors(X); return np.round([D[:,1].mean(),D[:,1].min(),(D[:,2]/D[:,1]).mean()],4)
report('NN dist mean/min, r2/r1 mean',nn)
def twonn(X):
    D,_=NearestNeighbors(n_neighbors=3).fit(X).kneighbors(X); mu=D[:,2]/D[:,1]; return round(n/np.log(mu).sum(),2)
report('TwoNN intrinsic dim',twonn)
# 5 row summaries
def rs(X):
    s=X.sum(1); return np.round([s.mean(),s.std(),stats.skew(s),stats.kurtosis(s)],3)
report('row-sum mean/std/skew/kurt',rs)
def rn(X):
    s=((X-.5)**2).sum(1); return np.round([s.mean(),s.std()],3)
report('row radius^2 mean/std',rn)
def fr(X):
    s=(X.sum(1))%1; return round(stats.kstest(s,'uniform').pvalue,4)
report('KS p of rowsum mod 1',fr)
# 6 duplicates/ digits
def dig(X):
    return np.round(stats.kstest((X*1e6)%1 .ravel() if False else ((X*1e6)%1).ravel(),'uniform').pvalue,4)
report('KS p low-order digits',dig)
# 7 row/col rank structure: sorted-within-row?
def mono(X): return np.round(np.abs(np.corrcoef(X.T)).sum()-d,2)
report('sum abs corr offdiag',mono)
