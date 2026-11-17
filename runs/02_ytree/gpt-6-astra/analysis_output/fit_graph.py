import os
os.environ['OPENBLAS_NUM_THREADS']='4'
import numpy as np,pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr,logsumexp,log_ndtr
from pathlib import Path
U=np.load('analysis_output/signal.npz')['U']; train=U[:3500];test=U[3500:]
# mixture of uniform finite line segments, convolved with circular Gaussian noise
initial=np.array([[.515,.515],[.44,.575],[.605,.54],[.505,.395]])
def components(theta,P):
 verts=theta[:8].reshape(4,2);sig=np.exp(theta[8]);A=verts[0];D=verts[1:]-A;L=np.linalg.norm(D,axis=1);E=D/L[:,None]
 Z=P[:,None,:]-A;t=np.einsum('nbd,bd->nb',np.broadcast_to(Z,(len(P),3,2)),E)
 perp2=np.maximum(0,((P-A)**2).sum(1)[:,None]-t*t)
 a=t/sig;b=(t-L)/sig
 hi=np.where(a+b>0,log_ndtr(-b),log_ndtr(a));lo=np.where(a+b>0,log_ndtr(-a),log_ndtr(b))
 logmass=hi+np.log(np.maximum(-np.expm1(lo-hi),1e-300))
 logs=-.5*perp2/sig**2-np.log(sig*np.sqrt(2*np.pi))-np.log(L)+logmass
 weights=np.exp(np.r_[theta[9:11],0]-logsumexp(np.r_[theta[9:11],0]));return logs+np.log(weights),verts,sig,weights

def fun(theta):return -logsumexp(components(theta,train)[0],axis=1).mean()
x0=np.r_[initial.ravel(),np.log(.004),0,0]
fit=minimize(fun,x0,method='L-BFGS-B',bounds=[(v-.025,v+.025) for v in initial.ravel()]+[(np.log(.0001),np.log(.02))]+[(-3,3)]*2,options={'maxiter':500,'ftol':1e-12,'gtol':1e-6})
logs,V,sig,weights=components(fit.x,U);np.savez('analysis_output/graph_fit.npz',theta=fit.x,vertices=V,sigma=sig,weights=weights)
def dist(P,V):
 A=V[0];D=V[1:]-A;t=np.clip((P-A)@D.T/(D*D).sum(1),0,1);Q=A+t[:,:,None]*D;return np.sqrt(((P[:,None,:]-Q)**2).sum(2).min(1))
rng=np.random.default_rng(493734);noise=rng.random((200000,2));dt=dist(train,V);dv=dist(test,V);dn=dist(noise,V)
msg=[f'Finite-segment Gaussian mixture fit success={fit.success}: {fit.message}',f'Y graph vertices (junction then tips): {V.tolist()}',f'Isotropic coordinate noise sigma {sig}; branch weights {weights}',f'Mean held-out log density {logsumexp(components(fit.x,test)[0],axis=1).mean()}; uniform baseline log density=0',f'Discovery distance quantiles (.5,.9,.95,.99): {np.quantile(dt,[.5,.9,.95,.99])}',f'Validation distance quantiles (.5,.9,.95,.99): {np.quantile(dv,[.5,.9,.95,.99])}',f'Uniform-noise distance quantiles (.01,.05,.5): {np.quantile(dn,[.01,.05,.5])}']
for radius in [.005,.01,.015,.02]:msg.append(f'Within distance {radius}: discovery {np.mean(dt<radius):.5f}, validation {np.mean(dv<radius):.5f}, uniform noise {np.mean(dn<radius):.6f}')
with Path('analysis_output/log.txt').open('a') as f:
 for s in msg: print(s);f.write(s+'\n')
