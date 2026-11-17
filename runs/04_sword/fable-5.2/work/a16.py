import numpy as np, pandas as pd, itertools
X=pd.read_csv('../X.csv').values; n,d=X.shape; x=lambda j:X[:,j-1]
res=[]
for c in itertools.product(range(-4,5),repeat=5):
    s=c[0]*x(2)+c[1]*x(4)+c[2]*x(15)+c[3]*x(20)+c[4]*x(28)
    res.append((abs(np.exp(2j*np.pi*(x(14)-s)).mean()),c))
res.sort(reverse=True); print(res[:5])
c=res[0][1]
v=(x(14)-(c[0]*x(2)+c[1]*x(4)+c[2]*x(15)+c[3]*x(20)+c[4]*x(28)))%1
print(np.histogram(v,bins=40,range=(0,1))[0])
