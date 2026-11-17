import numpy as np, pandas as pd, itertools
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
X=pd.read_csv('../X.csv').values; n,d=X.shape
x=lambda j:X[:,j-1]
res=[]
for c in itertools.product(range(-4,5),repeat=5):
    s=c[0]*x(9)+c[1]*x(12)+c[2]*x(25)+c[3]*x(27)+c[4]*x(29)
    res.append((abs(np.exp(2j*np.pi*(x(3)-s)).mean()),c))
res.sort(reverse=True); print(res[:5])
fig,ax=plt.subplots(2,3,figsize=(18,12))
ax[0,0].scatter(x(2),x(4),c=x(14),s=3,cmap='hsv');ax[0,0].set_title('x2,x4 colored x14')
r=np.hypot(x(2)-.5,x(4)-.5)
ax[0,1].scatter(r,x(14),s=2);ax[0,1].set_title('r vs x14')
ax[0,2].scatter(x(2),x(14),s=2,c=x(4));ax[0,2].set_title('x2 vs x14')
ax[1,0].scatter(abs(x(2)-.5),x(14),s=2)
ax[1,1].scatter((x(2)-.5)**2+(x(4)-.5)**2,x(14),s=2)
ax[1,2].scatter(np.arctan2(x(4)-.5,x(2)-.5),x(14),s=2)
plt.savefig('p_x14.png',dpi=70)
