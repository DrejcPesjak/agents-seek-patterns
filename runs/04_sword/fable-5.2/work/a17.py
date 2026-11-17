import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
X=pd.read_csv('../X.csv').values; n,d=X.shape; x=lambda j:X[:,j-1]
u=(x(3)-x(9)+x(12)+2*x(25)+2*x(27)-x(29))%1
v=(x(14)-2*x(2)-2*x(4)+x(15)+x(20)+x(28))%1
np.save('uv.npy',np.c_[u,v])
rng=np.random.default_rng(0);U=rng.random((n,30));y=lambda j:U[:,j-1]
un=(y(3)-y(9)+y(12)+2*y(25)+2*y(27)-y(29))%1; vn=(y(14)-2*y(2)-2*y(4)+y(15)+y(20)+y(28))%1
fig,ax=plt.subplots(1,2,figsize=(16,8))
ax[0].scatter(u,v,s=2);ax[0].set_title('data: u vs v');ax[0].set_xlim(0,1);ax[0].set_ylim(0,1);ax[0].grid(alpha=.3)
ax[1].scatter(un,vn,s=2,c='gray');ax[1].set_title('uniform noise through same map')
for a in ax:a.set_xlabel('u=(x3-x9+x12+2x25+2x27-x29) mod 1');a.set_ylabel('v=(x14-2x2-2x4+x15+x20+x28) mod 1');a.set_aspect('equal')
plt.savefig('p_uv.png',dpi=75)
