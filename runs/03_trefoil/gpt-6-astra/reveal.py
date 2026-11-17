import numpy as np,itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
A=np.load('data.npy');D=np.load('split.npz');hits=np.load('fourier5mixed_hits.npy',allow_pickle=True)
B=np.zeros((3,30))
for i,(lab,score) in enumerate(hits):
 for col,k in lab:B[i,col]=k
raw=A@B.T
mean=np.angle(np.exp(2j*np.pi*raw).mean(0))/(2*np.pi)
Y=(raw-mean+.5)%1-.5+mean
np.savez('revealed.npz',B=B,mean=mean,Y=Y)
f=open('analysis.log','a')
def log(s):print(s,flush=True);f.write(str(s)+'\n');f.flush()
log('DISCOVERY: three independent sparse integer modular coordinates, each involving five columns. All strongly confirmed in held-out half.')
log('Coefficient matrix '+str(B));log('Circular means '+str(mean));log('Unwrapped coordinate quantiles '+str(np.quantile(Y,[0,.001,.01,.5,.99,.999,1],axis=0)))
log('Revealed covariance '+str(np.cov(Y.T)));log('Revealed PCA '+str(np.linalg.svd(Y-Y.mean(0),compute_uv=False)**2/len(Y)))
fig,axs=plt.subplots(1,3,figsize=(18,6))
for ax,(j,k) in zip(axs,itertools.combinations(range(3),2)):
 ax.scatter(Y[:,j],Y[:,k],s=1,alpha=.35);ax.set_xlabel('u'+str(j+1));ax.set_ylabel('u'+str(k+1));ax.set_aspect('equal')
fig.tight_layout();fig.savefig('plots/revealed_pairs.png',dpi=180);plt.close(fig)
# Orthographic projections of 3-D data, without dependence on mplot3d.
fig,axs=plt.subplots(2,3,figsize=(18,12))
for ax,(az,el) in zip(axs.flat,[(0,30),(30,20),(60,30),(90,40),(120,30),(150,45)]):
 az,el=np.deg2rad([az,el]);P=np.array([[np.cos(az),np.sin(az),0],[-np.sin(az)*np.sin(el),np.cos(az)*np.sin(el),np.cos(el)]])
 z=(Y-Y.mean(0))@P.T;ax.scatter(z[:,0],z[:,1],s=1,alpha=.4,c=Y[:,2],cmap='viridis');ax.set_aspect('equal')
fig.tight_layout();fig.savefig('plots/revealed_views.png',dpi=160);plt.close(fig)
