import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
X=pd.read_csv('../X.csv').values; n,d=X.shape; exc=np.load('exc.npy'); x=lambda j:X[:,j-1]
np.set_printoptions(linewidth=250,precision=3,suppress=True)
u=(x(3)-x(9)+x(12)+2*x(25)+2*x(27)-x(29))%1
for t in (13,2):
    y=np.minimum(exc[:,t],3).astype(float)
    A=np.c_[X,u]
    m=HistGradientBoostingRegressor(max_depth=4,max_iter=200,learning_rate=0.05).fit(A[:6000],y[:6000])
    print('target',t+1,'test R2',m.score(A[6000:],y[6000:]))
    pi=permutation_importance(m,A[6000:],y[6000:],n_repeats=5,random_state=0).importances_mean
    print(pi)
# x14 conditional exc by bins of x2,x4
y=np.minimum(exc[:,13],3)
for j in (2,4,14):
    b=np.minimum((x(j)*10).astype(int),9); print('x%d bins mean exc14'%j,np.array([y[b==k].mean() for k in range(10)]), 'nonint frac',np.array([(exc[:,13]<0)[b==k].mean() for k in range(10)]))
b=np.minimum(((u-.3)/.35*10).astype(int),9);print('u bins',np.array([y[b==k].mean() for k in range(10)]))
