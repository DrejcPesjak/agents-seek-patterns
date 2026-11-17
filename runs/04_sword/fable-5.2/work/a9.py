import numpy as np, pandas as pd
X=pd.read_csv('../X.csv').values; n,d=X.shape
b=X.view(np.uint64)
np.set_printoptions(linewidth=250,precision=2,suppress=True)
for bit in range(6): print('bit',bit,((b>>np.uint64(bit))&np.uint64(1)).mean(0))
# is x * 2^53 integer?
for j in [0,2,3]:
    v=X[:,j]*2**53; print(j+1,'frac of 2^53 ints',np.mean(v==np.floor(v)), 'x*2^32 int',np.mean((X[:,j]*2**32)%1==0))
