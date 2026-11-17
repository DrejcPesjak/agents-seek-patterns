import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict
X=pd.read_csv('../X.csv').values; n,d=X.shape; exc=np.load('exc.npy')
np.set_printoptions(linewidth=250,precision=3,suppress=True)
for t in [2,13]:
    y=np.minimum(exc[:,t],3).astype(float)
    print('target x%d: corr of min(exc,3) with columns'%(t+1)); print(np.array([np.corrcoef(y,X[:,j])[0,1] for j in range(d)]))
    print(' corr with |x-0.5|'); print(np.array([np.corrcoef(y,abs(X[:,j]-.5))[0,1] for j in range(d)]))
    p=cross_val_predict(HistGradientBoostingRegressor(max_depth=4,max_iter=200,learning_rate=0.05),X,y,cv=4)
    print(' GBM R2',1-((y-p)**2).mean()/y.var())
    # real-coefficient scan
    A=np.arange(-12,12.001,0.02)
    for j in range(d):
        if j==t: continue
        v=np.abs(np.exp(2j*np.pi*(X[:,[t]]-X[:,[j]]*A[None,:])).mean(0))
        if v.max()>0.05: print('  scan col',j+1,'a=%.2f'%A[v.argmax()],'|m|=%.3f'%v.max())
