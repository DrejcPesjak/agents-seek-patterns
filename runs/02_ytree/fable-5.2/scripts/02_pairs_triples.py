import numpy as np, pandas as pd, itertools
X = pd.read_csv('X.csv').values
n,d = X.shape
# pairwise 2D histogram chi2 (10x10 bins): expected chi2 ~ 99 +- 14
B=10; Q=np.minimum((X*B).astype(int),B-1)
res=[]
for i in range(d):
    for j in range(i+1,d):
        H=np.bincount(Q[:,i]*B+Q[:,j],minlength=B*B); e=n/B/B
        res.append((((H-e)**2/e).sum(),i+1,j+1))
res.sort(reverse=True); print('pair chi2 top (null ~99±14, max of 435 ~ 145):', [(round(a,1),i,j) for a,i,j in res[:6]])
# Fourier characters, pairs with coefs up to 4
E={k:np.exp(2j*np.pi*k*X) for k in range(-4,5)}
out=[]
for i in range(d):
    for j in range(i+1,d):
        for a in range(1,5):
            for b in range(-4,5):
                if b==0: continue
                out.append((abs((E[a][:,i]*E[b][:,j]).mean()),i+1,j+1,a,b))
out.sort(reverse=True); print('pair chars top (null max ~', round(np.sqrt(np.log(len(out))/n),3),')', [(round(float(r),3),)+tuple(t) for r,*t in out[:6]])
# triples coefs in {-2..2}
out=[]
for i,j,k in itertools.combinations(range(d),3):
    for a in (1,2):
        P=E[a][:,i]
        for b in (-2,-1,1,2):
            PQ=P*E[b][:,j]
            M=np.abs(np.stack([E[c][:,k] for c in (-2,-1,1,2)])@PQ)/n
            for c,m in zip((-2,-1,1,2),M): out.append((m,i+1,j+1,k+1,a,b,c))
out.sort(reverse=True); print('triple chars top (null max ~', round(np.sqrt(np.log(len(out))/n),3),')'); [print((round(float(r),3),)+tuple(t)) for r,*t in out[:10]]
