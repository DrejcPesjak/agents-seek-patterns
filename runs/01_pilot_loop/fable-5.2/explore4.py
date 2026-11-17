import numpy as np, struct
from scipy import stats
X=np.loadtxt('example_X.csv',delimiter=',')
n,d=X.shape
rng=np.random.default_rng(2)
R=rng.random((n,d))
# A. high-harmonic pair relations a*x_i - b*x_j = const mod 1, a,b in 1..48 (both signs)
def hh(X,K=48):
    Z=[None]+[np.exp(2j*np.pi*k*X) for k in range(1,K+1)]
    best=(0,None); allmax=[]
    for a in range(1,K+1):
        for b in range(a,K+1):
            for W in (Z[b].conj(),Z[b]):
                M=np.abs(Z[a].T@W)/n; np.fill_diagonal(M,0)
                m=M.max(); allmax.append(m)
                if m>best[0]: best=(m,(a,b,np.unravel_index(M.argmax(),M.shape)))
    return best, np.round(np.percentile(allmax,[50,99]),4)
print('high-harm pair data',hh(X)); print('high-harm pair null',hh(R))
# also single-column harmonics (non-uniform fine structure)
def sh(X,K=200):
    return np.round(max(np.abs(np.exp(2j*np.pi*k*X).mean(0)).max() for k in range(1,K+1)),4)
print('single col harmonic max data',sh(X),'null',sh(R))
# B. torus / euclid NN distance quantiles
from sklearn.neighbors import NearestNeighbors
def q(X,torus=False):
    if torus:
        E=np.hstack([np.cos(2*np.pi*X),np.sin(2*np.pi*X)])
    else: E=X
    D,_=NearestNeighbors(n_neighbors=2).fit(E).kneighbors(E)
    return np.round(np.percentile(D[:,1],[0,1,10,50,90,100]),3)
print('euclid NN quantiles data',q(X),'null',q(R))
print('torus  NN quantiles data',q(X,True),'null',q(R,True))
# C. bit-level: trailing zero bits, exact duplicates, cross-cell equalities
u=np.unique(X.ravel()); print('distinct values',len(u),'of',X.size)
m=(X.ravel()*2**53); print('frac integer at 2^53',np.mean(m==np.round(m)),' at 2^32',np.mean((X.ravel()*2**32)%1==0), ' at 2^24',np.mean((X.ravel()*2**24)%1==0))
# D. consecutive-value lattice (RANDU-like) in row-major and col-major stream, coefs up to 10
for name,st in (('rowmajor',X.ravel()),('colmajor',X.T.ravel())):
    best=0
    for a in range(-10,11):
        for b in range(-10,11):
            if a==0 and b==0: continue
            v=abs(np.exp(2j*np.pi*(st[2:]+a*st[1:-1]+b*st[:-2])).mean()); best=max(best,v)
    print(name,'stream lattice max',round(best,4),'(noise ~',round(3/np.sqrt(st.size),4),')')
# E. per-row statistic distributions vs theory
for nm,f in (('mean',lambda A:A.mean(1)),('var',lambda A:A.var(1)),('max',lambda A:A.max(1)),('circmean R',lambda A:np.abs(np.exp(2j*np.pi*A).mean(1)))):
    print(nm,'KS data-vs-null p=',round(stats.ks_2samp(f(X),f(rng.random((20000,d)))).pvalue,4))
