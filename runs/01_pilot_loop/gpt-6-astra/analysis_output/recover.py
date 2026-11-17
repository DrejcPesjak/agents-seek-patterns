import os
os.environ['MPLCONFIGDIR']='/tmp/mpl-cache'
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
X=np.load('analysis_output/X.npy')
a=(X[:,3]-X[:,8]+X[:,19]+X[:,25]-X[:,29]-X[:,49]+.5)%1-.5
b=(X[:,18]-X[:,28]+X[:,30]+X[:,35]+X[:,37]+X[:,45]+.5)%1-.5
Y=np.c_[a,b];np.save('analysis_output/recovered_coordinates.npy',Y)
print('ranges',Y.min(0),Y.max(0))
fig,axs=plt.subplots(1,2,figsize=(13,6))
axs[0].scatter(a,b,s=3,alpha=.5);axs[0].set_aspect('equal');axs[0].set_xlabel('u');axs[0].set_ylabel('v')
axs[1].hist2d(a,b,bins=100);axs[1].set_aspect('equal');fig.tight_layout();fig.savefig('analysis_output/recovered_shape.png',dpi=160)
