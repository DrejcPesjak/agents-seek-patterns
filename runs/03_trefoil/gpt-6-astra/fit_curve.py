import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import eigsh
from scipy.interpolate import splprep,splev
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=np.load('revealed.npz');Y=D['Y']+1;n=len(Y)
f=open('analysis.log','a')
def log(s):print(s,flush=True);f.write(str(s)+'\n');f.flush()
# Spectral embedding of the local-neighbor graph discovers circular ordering.
t=cKDTree(Y);dist,idx=t.query(Y,k=21)
rows=np.repeat(np.arange(n),20);cols=idx[:,1:].ravel();scale=np.maximum(dist[:,10],.002)
w=np.exp(-dist[:,1:].ravel()**2/(scale[rows]*scale[cols]))
G=coo_matrix((w,(rows,cols)),shape=(n,n)).tocsr();G=(G+G.T)/2
ss=np.asarray(G.sum(axis=1)).ravel();N=diags(1/np.sqrt(ss));L=diags(np.ones(n))-N@G@N
vals,vecs=eigsh(L,k=8,sigma=-1e-6,which='LM');log('Laplacian eigenvalues '+str(vals))
V=vecs[:,1:3]/np.sqrt(ss[:,None]);theta=np.arctan2(V[:,1]/V[:,1].std(),V[:,0]/V[:,0].std())%(2*np.pi)
# Circular quantiles approximately undo sampling density / graph distortions.
rank=np.argsort(np.argsort(theta));phase=2*np.pi*(rank+.5)/n
np.savez('curve_order.npz',theta=theta,phase=phase,values=vals,vecs=vecs)
fig,axs=plt.subplots(1,3,figsize=(18,5))
for j,ax in enumerate(axs):ax.scatter(phase,Y[:,j],s=1);ax.set_xlabel('estimated phase');ax.set_ylabel(f'u{j+1}')
fig.tight_layout();fig.savefig('plots/curve_order.png',dpi=150)
for H in [3,5,8,12,20,30]:
 F=np.column_stack([np.ones(n)]+[fn(k*phase) for k in range(1,H+1) for fn in [np.cos,np.sin]])
 B=np.linalg.lstsq(F,Y,rcond=None)[0];err=Y-F@B
 log(f'Initial phase Fourier harmonics={H}: residual RMS {np.sqrt(np.mean(err**2)):.6f}, per-coordinate {np.std(err,axis=0)}')
# Seed a regularized closed smooth curve then alternate closest-point assignments and Fourier fitting.
H=12;ph=phase.copy();train=np.load('split.npz')['train'];# deterministic training split same as original
ix=np.random.default_rng(321).permutation(n);trainix=ix[:4000];testix=ix[4000:]
def features(t,H):return np.column_stack([np.ones(len(t))]+[fn(k*t) for k in range(1,H+1) for fn in [np.cos,np.sin]])
B=np.linalg.lstsq(features(ph[trainix],H),Y[trainix],rcond=None)[0]
for it in range(15):
 grid=np.arange(12000)*2*np.pi/12000;curve=features(grid,H)@B
 dd,ids=cKDTree(curve).query(Y);ph=grid[ids]
 # Fixed-tangent local Newton correction for subgrid phase.
 deriv=np.column_stack([np.zeros(len(ph))]+[v for k in range(1,H+1) for v in [-k*np.sin(k*ph),k*np.cos(k*ph)]])@B
 resid=Y-features(ph,H)@B
 ph=(ph+np.sum(resid*deriv,axis=1)/np.sum(deriv**2,axis=1))%(2*np.pi)
 F=features(ph[trainix],H);reg=np.diag([0]+[k**4 for k in range(1,H+1) for _ in range(2)])*.01
 B=np.linalg.solve(F.T@F+reg,F.T@Y[trainix])
 if it in [0,4,9,14]:
  err=Y-features(ph,H)@B;log(f'Principal curve iter {it}: train distance RMS {np.sqrt(np.mean(np.sum(err[trainix]**2,1))):.6f}; test {np.sqrt(np.mean(np.sum(err[testix]**2,1))):.6f}')
np.savez('curve_fit.npz',B=B,H=H,phase=ph,trainix=trainix,testix=testix)
grid=np.arange(200)*2*np.pi/200;curve=features(grid,H)@B
fig,axs=plt.subplots(1,3,figsize=(18,5))
for ax,(j,k) in zip(axs,[(0,1),(0,2),(1,2)]):
 ax.scatter(Y[testix,j],Y[testix,k],s=1,alpha=.3);ax.plot(np.r_[curve[:,j],curve[0,j]],np.r_[curve[:,k],curve[0,k]],'r-',lw=1);ax.set_aspect('equal');ax.set_xlabel(f'u{j+1}');ax.set_ylabel(f'u{k+1}')
fig.tight_layout();fig.savefig('plots/fitted_curve.png',dpi=180)
