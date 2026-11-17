import numpy as np,time
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
D=np.load('revealed.npz');Y=D['Y']+1;C=np.load('curve_fit.npz');ix=C['trainix'];jx=C['testix'];initial=C['phase'];n=len(ix)
f=open('analysis.log','a')
def log(s):print(s,flush=True);f.write(str(s)+'\n');f.flush()
def feat(t,H):return np.column_stack([np.ones(len(t))]+[fn(k*t) for k in range(1,H+1) for fn in [np.cos,np.sin]])
def dfeat(t,H):return np.column_stack([np.zeros(len(t))]+[v for k in range(1,H+1) for v in [-k*np.sin(k*t),k*np.cos(k*t)]])
for H in [3,5,8]:
 p=1+2*H;t=initial[ix].copy();B=np.linalg.lstsq(feat(t,H),Y[ix],rcond=None)[0]
 # Optimize phases and global Fourier coefficients jointly using an analytic sparse Jacobian.
 def fun(z):
  B=z[:p*3].reshape(p,3);t=z[p*3:];return (feat(t,H)@B-Y[ix]).ravel()
 def jac(z):
  B=z[:p*3].reshape(p,3);t=z[p*3:];F=feat(t,H);v=dfeat(t,H)@B
  rows=np.repeat(np.arange(3*n),p);cols=np.tile(np.arange(p)*3,3*n)+np.repeat(np.tile(np.arange(3),n),p)
  vals=np.repeat(F,3,axis=0).ravel()
  rows=np.r_[rows,np.arange(3*n)];cols=np.r_[cols,p*3+np.repeat(np.arange(n),3)];vals=np.r_[vals,v.ravel()]
  return coo_matrix((vals,(rows,cols)),shape=(3*n,p*3+n)).tocsr()
 t0=time.time();res=least_squares(fun,np.r_[B.ravel(),t],jac=jac,method='trf',ftol=1e-9,max_nfev=200,verbose=0)
 B=res.x[:p*3].reshape(p,3);t=res.x[p*3:]
 grid=np.arange(100000)*2*np.pi/100000;curve=feat(grid,H)@B;dd,ids=cKDTree(curve).query(Y[jx]);ph=grid[ids]
 der=dfeat(ph,H)@B;ph=ph+np.sum((Y[jx]-feat(ph,H)@B)*der,axis=1)/np.sum(der**2,axis=1)
 testerr=Y[jx]-feat(ph,H)@B
 log(f'Joint Fourier fit H={H}: train Euclidean RMS {np.sqrt(np.mean(fun(res.x).reshape(-1,3)**2)*3):.7f}; test RMS {np.sqrt(np.mean(testerr**2)*3):.7f}; evaluations {res.nfev}, elapsed {time.time()-t0:.1f}s')
 log(f'Joint Fourier H={H} harmonic amplitudes '+str(np.linalg.norm(B[1:].reshape(H,2,3),axis=(1,2))))
 np.savez(f'fourier_curve_H{H}.npz',B=B,H=H,trainphase=t,testphase=ph,trainix=ix,testix=jx)
