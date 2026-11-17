import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
X=pd.read_csv('../X.csv').values; n,d=X.shape
v=X*2**53; nonint=(v!=np.floor(v)); print('nonint per col',nonint.sum(0))
b=X.view(np.uint64)
# trailing zero count relative to expectation: for uniform k*2^-53 values, x in [2^-e-1,2^-e) has e trailing zeros at least
m,e=np.frexp(X); tz=np.array([[(int(q)&-int(q)).bit_length()-1 if q else 64 for q in r] for r in b])
exc=tz-(-e)  # extra trailing zeros beyond min forced
print('mean excess tz per col',np.round(exc.mean(0),2))
for j in [0,2,13]:
    print(j+1,'excess dist',np.bincount(np.clip(exc[:,j],-3,8)+3))
fig,ax=plt.subplots(1,3,figsize=(18,6))
ax[0].scatter(X[:,2],X[:,13],s=2);ax[0].set_title('x3 vs x14')
ax[1].scatter(X[:,2],X[:,13],s=2,c=exc[:,2]<0);ax[1].set_title('colored by x3 odd bits')
ax[2].scatter((X[:,2]+X[:,13])%1,(X[:,2]-X[:,13])%1,s=2)
plt.savefig('p_x3_x14.png',dpi=80)
np.save('exc.npy',exc)
