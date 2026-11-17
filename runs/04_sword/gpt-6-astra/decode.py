import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
X=np.load('analysis_outputs/X.npy')
k1=np.zeros(30);k1[[2,8,11,24,26,28]]=[1,-1,1,2,2,-1]
k2=np.zeros(30);k2[[1,3,13,14,19,27]]=[2,2,-1,-1,-1,-1]
Y=np.c_[X@k1%1,X@k2%1];np.save('analysis_outputs/decoded.npy',Y)
print('range',Y.min(0),Y.max(0),'mean',Y.mean(0),'std',Y.std(0));print('quantiles',np.quantile(Y,[0,.01,.1,.25,.5,.75,.9,.99,1],axis=0))
fig,axs=plt.subplots(1,3,figsize=(16,5))
axs[0].scatter(Y[:,0],Y[:,1],s=2,alpha=.4);axs[0].set(xlim=(0,1),ylim=(0,1),aspect='equal',title='Decoded modulo-1 coordinates')
for j,ax in enumerate(axs[1:]):ax.hist(Y[:,j],bins=100);ax.set_title('Decoded coordinate '+str(j+1))
fig.tight_layout();fig.savefig('analysis_outputs/decoded_initial.png',dpi=170)
print('complex means',np.exp(2j*np.pi*Y).mean(0))
