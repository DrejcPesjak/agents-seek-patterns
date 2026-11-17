import numpy as np, pandas as pd
X=pd.read_csv('../X.csv').values; x=lambda j:X[:,j-1]
u=(x(3)-x(9)+x(12)+2*x(25)+2*x(27)-x(29))%1
print(np.histogram(u,bins=40,range=(0,1))[0])
print('circ mean',np.angle(np.exp(2j*np.pi*u).mean())/(2*np.pi)%1)
