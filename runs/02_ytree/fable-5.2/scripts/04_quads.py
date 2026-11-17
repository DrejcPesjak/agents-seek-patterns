import numpy as np, pandas as pd, itertools
X = pd.read_csv('X.csv').values
n=len(X)
for cols in ([10,23,25,30],[1,6,11,24]):
    P=X[:,[c-1 for c in cols]]
    print('==',cols)
    out=[]
    for co in itertools.product(range(-4,5),repeat=4):
        if co==(0,0,0,0) or next(c for c in co if c!=0)<0: continue
        out.append((abs(np.exp(2j*np.pi*(P@np.array(co))).mean()),co))
    out.sort(reverse=True)
    for r,co in out[:12]: print(round(float(r),3),co)
    # is relation present in real (non-mod) coordinates? linear regression of each on the others w/ quadratic terms
    Z=P-0.5
    F=np.column_stack([Z[:,i]*Z[:,j] for i in range(4) for j in range(i,4)]+[np.prod(Z,1)])
    print('corr of prod(x-.5) moments: E[prod]=',np.prod(Z,1).mean(), 'null sd', (1/12)**2/np.sqrt(n))
