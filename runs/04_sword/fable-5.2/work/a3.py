import numpy as np, pandas as pd, itertools
X=pd.read_csv('../X.csv').values; n,d=X.shape
def scan(B,r):
    Q=np.minimum((X*B).astype(int),B-1)
    out=[]
    for c in itertools.combinations(range(d),r):
        idx=np.zeros(n,int)
        for j in c: idx=idx*B+Q[:,j]
        h=np.bincount(idx,minlength=B**r); e=n/B**r
        chi=((h-e)**2/e).sum(); dof=B**r-1
        out.append(((chi-dof)/np.sqrt(2*dof),c))
    out.sort(reverse=True); return out
for B,r in [(16,2),(8,3),(5,3),(4,4),(3,4)]:
    o=scan(B,r); z=np.array([a for a,_ in o])
    print(B,r,'top',[(round(a,1),tuple(i+1 for i in c)) for a,c in o[:6]],'median',np.median(z),'min',z.min())
