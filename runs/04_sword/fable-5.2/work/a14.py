import numpy as np, pandas as pd, itertools
X=pd.read_csv('../X.csv').values; n,d=X.shape; x=lambda j:X[:,j-1]
res=[]
for a,b in itertools.product(range(-6,7),repeat=2):
    res.append((abs(np.exp(2j*np.pi*(x(14)+a*x(2)+b*x(4))).mean()),a,b))
res.sort(reverse=True); print(res[:4])
# with one/two extra columns
others=[j for j in range(1,31) if j not in (2,4,14)]
res=[]
for a,b in itertools.product(range(-3,4),repeat=2):
    base=np.exp(2j*np.pi*(x(14)+a*x(2)+b*x(4)))
    for j in others:
        for c in range(-3,4):
            if c==0: continue
            B=base*np.exp(2j*np.pi*c*x(j))
            res.append((abs(B.mean()),a,b,(j,c)))
            for k in others:
                if k<=j: continue
                for e in (-2,-1,1,2):
                    res.append((abs((B*np.exp(2j*np.pi*e*x(k))).mean()),a,b,(j,c),(k,e)))
res.sort(reverse=True,key=lambda t:t[0]); print(res[:6])
