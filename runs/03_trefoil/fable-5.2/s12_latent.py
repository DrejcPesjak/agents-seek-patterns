import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
X=pd.read_csv('X.csv').values; n,d=X.shape
x=lambda j:X[:,j-1]
U=np.c_[x(3)-x(6)+2*x(8)-2*x(10)-x(29), x(4)-2*x(13)-x(19)+2*x(22)+x(30), x(2)+2*x(12)+x(16)+x(24)+2*x(27)]%1
np.save('U.npy',U)
for k in range(3):
    h=np.histogram(U[:,k],50,(0,1))[0]; print('u%d'%(k+1),'range',U[:,k].min().round(3),U[:,k].max().round(3),'empty bins',(h==0).sum(), 'circ mean',(np.angle(np.exp(2j*np.pi*U[:,k]).mean())/(2*np.pi)%1).round(4))
fig,ax=plt.subplots(2,3,figsize=(16,10))
for k,(i,j) in enumerate([(0,1),(0,2),(1,2)]):
    ax[0,k].plot(U[:,i],U[:,j],'.',ms=1.5); ax[0,k].set_xlabel(f'u{i+1}'); ax[0,k].set_ylabel(f'u{j+1}'); ax[0,k].set_xlim(0,1); ax[0,k].set_ylim(0,1); ax[0,k].set_aspect('equal')
    ax[1,k].hist(U[:,k],200,range=(0,1)); ax[1,k].set_title(f'u{k+1}')
plt.tight_layout(); plt.savefig('plot_latent_raw.png',dpi=100)
