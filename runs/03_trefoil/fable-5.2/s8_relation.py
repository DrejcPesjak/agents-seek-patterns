import numpy as np, pandas as pd
from s7_lll import test, check
X=pd.read_csv('X.csv').values; n,d=X.shape
found={}
for seed in range(1,9):
    for W in (5,10,20):
        for nm,a in test(X,W=W,seed=seed):
            if abs(a).sum()>0:
                if a[np.nonzero(a)[0][0]]<0: a=-a
                found[tuple(a)]=check(X,a)
for a,s in sorted(found.items(),key=lambda t:-t[1]):
    a=np.array(a); print(round(s,1),{int(i)+1:int(a[i]) for i in np.nonzero(a)[0]})
