import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
Y=np.load('analysis_output/recovered_coordinates.npy');rng=np.random.default_rng(92);Y=Y[rng.choice(len(Y),500,replace=False)]
c=Y.mean(0);theta=np.arctan2(Y[:,1]-c[1],Y[:,0]-c[0]);n=len(Y)
for m in [2,3]:
 def design(t):return np.column_stack([np.ones(len(t))]+[f(k*t) for k in range(1,m+1) for f in [np.cos,np.sin]])
 def deriv(t):return np.column_stack([np.zeros(len(t))]+[f(k*t)*s*k for k in range(1,m+1) for f,s in [(np.sin,-1),(np.cos,1)]])
 q=2*m+1;coef=np.linalg.lstsq(design(theta),Y,rcond=None)[0]
 def fun(p):return (design(p[:n])@p[n:].reshape(q,2)-Y).ravel()
 def jac(p):
  A=design(p[:n]);C=p[n:].reshape(q,2);dt=deriv(p[:n])@C
  row=np.r_[np.arange(2*n),np.repeat(np.arange(2*n),q)]
  col=np.r_[np.repeat(np.arange(n),2),np.tile(np.arange(q)*2,2*n)+n+np.repeat(np.arange(2*n)%2,q)]
  data=np.r_[dt.ravel(),np.repeat(A,2,axis=0).ravel()]
  return coo_matrix((data,(row,col)),shape=(2*n,n+q*2)).tocsr()
 fit=least_squares(fun,np.r_[theta,coef.ravel()],jac=jac,ftol=1e-14,xtol=1e-14,gtol=1e-14,max_nfev=1000)
 print('harmonics',m,'status',fit.message,'nfev',fit.nfev,'rms',np.sqrt(np.mean(fit.fun**2)),'max',abs(fit.fun).max(),flush=True)
 print(fit.x[n:].reshape(q,2),flush=True)
 np.savez(f'analysis_output/parametric_{m}.npz',coef=fit.x[n:].reshape(q,2),theta=fit.x[:n])
