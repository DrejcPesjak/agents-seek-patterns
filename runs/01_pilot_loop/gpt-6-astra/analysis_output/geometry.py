import os
os.environ['MPLCONFIGDIR']='/tmp/mpl-cache'
import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline
Y=np.load('analysis_output/recovered_coordinates.npy');rng=np.random.default_rng(862);ids=rng.permutation(len(Y));tr=Y[ids[:2800]];te=Y[ids[2800:]]
c=(Y.min(0)+Y.max(0))/2
print('center bbox',c)
# Fit radial Fourier series, report holdout Euclidean errors
for m in [4,8,12,16,24,32,48,64]:
 def design(t):return np.column_stack([np.ones(len(t))]+[f(k*t) for k in range(1,m+1) for f in [np.cos,np.sin]])
 rt=np.linalg.norm(tr-c,axis=1);tt=np.arctan2(tr[:,1]-c[1],tr[:,0]-c[0]);coef=np.linalg.lstsq(design(tt),rt,rcond=None)[0]
 theta=np.arctan2(te[:,1]-c[1],te[:,0]-c[0]);err=abs(np.linalg.norm(te-c,axis=1)-design(theta)@coef)
 print('radial Fourier',m,'holdout q50,95,max',np.quantile(err,[.5,.95,1]))
# Test whether a simple low degree implicit polynomial describes the curve.
T=(tr-c)/.27;E=(te-c)/.27
for deg in [2,3,4,5,6,8,10]:
 powers=[(i,j) for i in range(deg+1) for j in range(deg+1-i)]
 A=np.column_stack([T[:,0]**i*T[:,1]**j for i,j in powers]); _,s,V=np.linalg.svd(A,full_matrices=False);v=V[-1]
 def errdist(P):
  val=sum(a*P[:,0]**i*P[:,1]**j for a,(i,j) in zip(v,powers))
  dx=sum(a*i*P[:,0]**(i-1)*P[:,1]**j for a,(i,j) in zip(v,powers) if i)
  dy=sum(a*j*P[:,0]**i*P[:,1]**(j-1) for a,(i,j) in zip(v,powers) if j)
  return abs(val)/np.sqrt(dx*dx+dy*dy)*.27
 print('implicit',deg,'sing ratio',s[-1]/s[0],'holdout',np.quantile(errdist(E),[.5,.95,1]))
# periodic radial cubic spline with independent validation
theta=np.arctan2(tr[:,1]-c[1],tr[:,0]-c[0]);r=np.linalg.norm(tr-c,axis=1);order=np.argsort(theta);theta=theta[order];r=r[order]
spl=CubicSpline(np.r_[theta,theta[0]+2*np.pi],np.r_[r,r[0]],bc_type='periodic')
tt=np.arctan2(te[:,1]-c[1],te[:,0]-c[0]);pred=spl(tt);err=abs(pred-np.linalg.norm(te-c,axis=1));print('periodic spline holdout',np.quantile(err,[.5,.9,.95,.99,1]))
np.savez('analysis_output/radial_spline.npz',theta=theta,r=r,center=c,train=ids[:2800],test=ids[2800:])
