import numpy as np, itertools
from scipy import stats
X=np.loadtxt('example_X.csv',delimiter=',')
n,d=X.shape
rng=np.random.default_rng(1)
R=rng.random((n,d))
# A. pairwise chi2 top pairs
def chis(X,nb=8):
    B=np.minimum((X*nb).astype(int),nb-1); out={}
    for i in range(d):
        for j in range(i+1,d):
            H=np.zeros((nb,nb)); np.add.at(H,(B[:,i],B[:,j]),1)
            out[(i,j)]=((H-n/nb**2)**2/(n/nb**2)).sum()
    return out
for nb in (4,8,16):
    c=chis(X,nb); top=sorted(c.items(),key=lambda t:-t[1])[:5]
    nulls=[max(chis(rng.random((n,d)),nb).values()) for _ in range(5)]
    print('nb',nb,'top pairs',[(k,round(v,1)) for k,v in top],'null maxes',np.round(nulls,1))
# B. triple / quad integer relations mod 1 via Fourier
def trip(X,coefs=(1,2)):
    best=(0,None)
    Z={k:np.exp(2j*np.pi*k*X) for k in (-2,-1,1,2)}
    for a in (1,2):
        for b in (-2,-1,1,2):
            # pair products for all i<j : n x d x d too big? 4000*2500 complex ok (160MB)
            P=(Z[a][:,:,None]*Z[b][:,None,:]).reshape(n,d*d)
            for c in (-2,-1,1,2):
                M=np.abs(P.T@Z[c])/n   # (d*d, d)
                M=M.reshape(d,d,d)
                for i in range(d): M[i,i,:]=0; M[i,:,i]=0; M[:,i,i]=0
                idx=np.unravel_index(M.argmax(),M.shape)
                if M.max()>best[0]: best=(M.max(),(a,b,c,idx))
    return best
print('triple fourier max data',trip(X)); print('triple fourier max null',trip(R))
# C. along-row structure (columns as an ordered sequence)
def along(X):
    Xc=X-X.mean(1,keepdims=True)
    acs=[(Xc[:,:-k]*Xc[:,k:]).mean()/Xc.var() for k in (1,2,3,4,5)]
    F=np.abs(np.fft.rfft(Xc,axis=1))**2
    return np.round(acs,4), np.round(F.mean(0)[1:].max()/F.mean(0)[1:].mean(),3)
print('along-row acf / spectrum peak ratio data',along(X)); print('null',along(R))
# D. row-level scalar stats vs null: min,max,range, sorted gaps
def rowstats(X):
    S=np.sort(X,1)
    return np.round([S[:,0].mean(),S[:,-1].mean(),np.diff(S,axis=1).max(1).mean(),np.diff(S,axis=1).min(1).mean()],5)
print('row min/max/maxgap/mingap data',rowstats(X),'null',rowstats(R))
# E. Gaussianized PCA + kurtosis of projections (ICA-ish)
from scipy.special import ndtri
G=ndtri(np.clip(X,1e-9,1-1e-9)); s=np.linalg.svd(G-G.mean(0),compute_uv=False)
G0=ndtri(R); s0=np.linalg.svd(G0-G0.mean(0),compute_uv=False)
print('gaussianized sv top/bot data',np.round(s[:3],2),np.round(s[-3:],2),'null',np.round(s0[:3],2),np.round(s0[-3:],2))
