import numpy as np, pandas as pd, itertools
from scipy.spatial import cKDTree
from multiprocessing import Pool
X = pd.read_csv('X.csv').values
n,d = X.shape
def chi(args):
    cols,B=args
    Q=np.minimum((X[:,cols]*B).astype(int),B-1)
    idx=np.zeros(n,int)
    for c in range(len(cols)): idx=idx*B+Q[:,c]
    H=np.bincount(idx,minlength=B**len(cols)); e=n/B**len(cols)
    return ((H-e)**2/e).sum()
def knn(cols):
    P=X[:,cols]
    t=cKDTree(P,boxsize=1.0)
    dd,_=t.query(P,k=2)
    return np.log(dd[:,1]).mean()
if __name__=='__main__':
    with Pool(14) as p:
        for m,B in [(3,6),(3,4),(4,4),(4,3)]:
            combos=list(itertools.combinations(range(d),m))
            r=np.array(p.map(chi,[(list(c),B) for c in combos],chunksize=200))
            z=(r-r.mean())/r.std(); o=np.argsort(-z)[:6]
            print(f'chi2 m={m} B={B} df={B**m-1} mean={r.mean():.1f} sd={r.std():.1f} top:',[(tuple(np.array(combos[i])+1),round(float(z[i]),2)) for i in o])
        for m in (2,3,4):
            combos=list(itertools.combinations(range(d),m))
            r=np.array(p.map(knn,[list(c) for c in combos],chunksize=100))
            z=(r-r.mean())/r.std(); o=np.argsort(z)[:6]
            print(f'knn m={m} mean={r.mean():.4f} sd={r.std():.4f} most-negative z:',[(tuple(np.array(combos[i])+1),round(float(z[i]),2)) for i in o], 'max z', z.max().round(2))
