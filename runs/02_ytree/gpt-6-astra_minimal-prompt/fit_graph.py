import numpy as np, scipy.optimize as so
from scipy.special import ndtr,logsumexp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
Y=np.load('coordinates.npy');rng=np.random.default_rng(830);idx=rng.permutation(len(Y));train=Y[idx[:2500]];test=Y[idx[2500:]]
def unpack(p):return p[:8].reshape(4,2),np.exp(p[8]),np.exp(p[9:12]-logsumexp(p[9:12]))
def logdensity(p,Y):
 P,s,w=unpack(p);A=P[0];D=P[1:]-A;L=np.linalg.norm(D,axis=1);U=D/L[:,None];Q=Y[:,None,:]-A;t=np.einsum('nki,ki->nk',Q,U);perp=np.maximum((Q*Q).sum(2)-t*t,0)
 integ=np.maximum(ndtr(t/s)-ndtr((t-L)/s),1e-300)
 return logsumexp(np.log(w/L)[None,:]-.5*np.log(2*np.pi)-np.log(s)-perp/(2*s*s)+np.log(integ),axis=1)
def fit(Y,start):
 return so.minimize(lambda p:-logdensity(p,Y).mean(),start,method='L-BFGS-B',bounds=[(v-.025,v+.025) for v in p0[:8]]+[(np.log(.0001),np.log(.03))]+[(-4,4)]*3,options={'maxiter':1000,'ftol':1e-13,'gtol':1e-7})
p0=np.r_[np.array([[.515,.51],[.435,.575],[.605,.54],[.505,.4]]).ravel(),np.log(.004),np.zeros(3)]
r=fit(train,p0);P,s,w=unpack(r.x)
print('Train fit',r.success,r.message,'vertices',P,'sigma',s,'weights',w,'heldout NLL',-logdensity(r.x,test).mean(),flush=True)
np.savez('graph_train_fit.npz',p=r.x,train_indices=idx[:2500],test_indices=idx[2500:])
rall=fit(Y,r.x);P,s,w=unpack(rall.x)
print('Full fit',rall.success,'vertices',P,'sigma',s,'weights',w,flush=True)
np.savez('graph_fit.npz',p=rall.x,P=P,sigma=s,weights=w)
fig,ax=plt.subplots(1,2,figsize=(14,6))
for a,Z,title in zip(ax,[train,test],['Training rows','Held-out rows']):
 a.scatter(*Z.T,s=2,alpha=.35,label='Rows')
 for e in unpack(r.x)[0][1:]:a.plot(*np.array([unpack(r.x)[0][0],e]).T,color='crimson',lw=2)
 a.scatter(*unpack(r.x)[0].T,color='black',s=20);a.set_aspect('equal');a.set(xlabel='u',ylabel='v',title=title);a.grid(alpha=.2)
fig.tight_layout();fig.savefig('05_fitted_Y_graph.png')
