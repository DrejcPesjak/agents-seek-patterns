import numpy as np,time
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
D=np.load('revealed.npz');Y=D['Y']+1;C=np.load('fourier_curve_H5.npz');trainix=C['trainix'];testix=C['testix'];initial=C['trainphase']
f=open('analysis.log','a')
def log(s):print(s,flush=True);f.write(str(s)+'\n');f.flush()
def feat(t):
 r=2+np.cos(3*t)
 return np.column_stack([np.ones(len(t)),r*np.cos(2*t),r*np.sin(2*t),np.sin(3*t)])
def dfeat(t):
 r=2+np.cos(3*t);dr=-3*np.sin(3*t)
 return np.column_stack([np.zeros(len(t)),dr*np.cos(2*t)-2*r*np.sin(2*t),dr*np.sin(2*t)+2*r*np.cos(2*t),3*np.cos(3*t)])
def fit(ix,t,max_nfev):
 n=len(ix);p=4;B=np.linalg.lstsq(feat(t),Y[ix],rcond=None)[0]
 def fun(z):return (feat(z[12:])@z[:12].reshape(4,3)-Y[ix]).ravel()
 def jac(z):
  B=z[:12].reshape(4,3);t=z[12:];F=feat(t);v=dfeat(t)@B
  rows=np.repeat(np.arange(3*n),p);cols=np.tile(np.arange(p)*3,3*n)+np.repeat(np.tile(np.arange(3),n),p)
  vals=np.repeat(F,3,axis=0).ravel()
  return coo_matrix((np.r_[vals,v.ravel()],(np.r_[rows,np.arange(3*n)],np.r_[cols,12+np.repeat(np.arange(n),3)])),shape=(3*n,12+n)).tocsr()
 res=least_squares(fun,np.r_[B.ravel(),t],jac=jac,method='trf',ftol=1e-10,xtol=1e-10,gtol=1e-9,max_nfev=max_nfev)
 return np.sqrt(np.mean(fun(res.x)**2)*3),res.x[:12].reshape(4,3),res.x[12:],res.nfev
best=None
for delta in np.arange(12)*2*np.pi/12:
 score,B,t,nfev=fit(trainix[::4],initial[::4]+delta,200)
 log(f'Affine standard trefoil initial phase shift {delta:.3f}: train RMS {score:.7f}, evaluations {nfev}')
 if best is None or score<best[0]:best=(score,B,t,delta)
score,B,t,nfev=fit(trainix,initial+best[3],500)
grid=np.arange(100000)*2*np.pi/100000;curve=feat(grid)@B;dd,ids=cKDTree(curve).query(Y[testix]);ph=grid[ids]
for it in range(3):
 der=dfeat(ph)@B;ph=ph+np.sum((Y[testix]-feat(ph)@B)*der,axis=1)/np.sum(der**2,axis=1)
err=Y[testix]-feat(ph)@B
log(f'Affine standard trefoil FINAL train RMS {score:.7f}; test RMS {np.sqrt(np.mean(err**2)*3):.7f}')
log('Affine trefoil coefficients [center; standard x basis; standard y basis; standard z basis]:\n'+str(B))
log('Linear matrix singular values '+str(np.linalg.svd(B[1:],compute_uv=False)))
np.savez('trefoil_fit.npz',B=B,trainphase=t,testphase=ph,trainix=trainix,testix=testix)
