import numpy as np, json
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
U=np.load('U.npy'); n=len(U); rng=np.random.default_rng(0)
perm=rng.permutation(n); tr,te=U[perm[:5000]],U[perm[5000:]]
K=6
def design(t): return np.column_stack([np.ones_like(t)]+[f(k*t) for k in range(1,K+1) for f in (np.cos,np.sin)])
tt=np.linspace(0,2*np.pi,4000,endpoint=False); Dt=design(tt)
def fit(P,init,iters=60):
    C=init
    for _ in range(iters):
        t=tt[cKDTree(Dt@C).query(P)[1]]
        C=np.linalg.lstsq(design(t),P,rcond=None)[0]
    dist=cKDTree(Dt@C).query(P)[0]
    return C,dist
best=None
c0=tr.mean(0)
for s2 in (1,-1):
  for s1 in (1,-1):
    for rot in np.linspace(0,2*np.pi/3,6,endpoint=False):
        t=tt+rot
        base=np.column_stack([s1*0.115*np.sin(3*tt), 0.08*(np.sin(t)+2*np.sin(2*t)), s2*0.08*(np.cos(t)-2*np.cos(2*t))])+c0
        C0=np.linalg.lstsq(Dt,base,rcond=None)[0]
        C,dist=fit(tr,C0)
        if best is None or dist.mean()<best[1].mean(): best=(C,dist,(s1,s2,rot))
C,dist,cfg=best
print('best init',cfg,'train mean dist',dist.mean(),'rms',np.sqrt((dist**2).mean()))
curve=Dt@C; tree=cKDTree(curve)
dte,ite=tree.query(te); print('held-out mean dist',dte.mean(),'rms',np.sqrt((dte**2).mean()),'q99',np.quantile(dte,.99),'max',dte.max())
N=rng.random((20000,3)); print('uniform-noise mean dist',tree.query(N)[0].mean(),'frac noise within q99:',(tree.query(N)[0]<np.quantile(dte,.99)).mean())
# coverage along the curve (is the whole knot populated uniformly?)
h=np.histogram(tt[tree.query(U)[1]],40,(0,2*np.pi))[0]; print('occupancy per 1/40 of t: min',h.min(),'max',h.max())
# residual components: isotropic noise? per-axis std
res=U-curve[tree.query(U)[1]]; print('residual std per axis',res.std(0).round(4))
print('Fourier coefs (rows: 1,cos t,sin t,cos2t,sin2t,...):'); print(np.round(C,4))
# refit on all data, emit 200-point polyline
C,_=fit(U,C); np.save('C.npy',C)
M=200; tp=np.linspace(0,2*np.pi,M,endpoint=False); pts=design(tp)@C
# polyline error vs smooth curve
def seg_dist(P,A,B):
    d=B-A; s=np.clip(((P[:,None,:]-A[None])*d[None]).sum(-1)/(d*d).sum(-1)[None],0,1)
    return np.linalg.norm(P[:,None,:]-(A[None]+s[...,None]*d[None]),axis=-1).min(1)
A=pts;B=np.roll(pts,-1,0)
print('polyline: max dev of smooth curve from polyline',seg_dist(design(tt)@C,A,B).max(),' data mean dist to polyline',seg_dist(U,A,B).mean())
print('pts range',pts.min(0).round(3),pts.max(0).round(3))
np.save('pts.npy',pts)
fig=plt.figure(figsize=(17,5.5))
for k,(el,az) in enumerate([(20,35),(90,-90),(10,120)]):
    ax=fig.add_subplot(1,3,k+1,projection='3d'); ax.scatter(*U.T,s=1,alpha=.25); ax.plot(*np.vstack([pts,pts[:1]]).T,'r-',lw=1.5)
    ax.view_init(el,az); ax.set_xlabel('u1');ax.set_ylabel('u2');ax.set_zlabel('u3')
plt.suptitle('Latent coordinates (blue: 8000 rows) and fitted 200-point closed polyline (red)'); plt.tight_layout(); plt.savefig('plot_knot_3d.png',dpi=100)
fig,ax=plt.subplots(1,2,figsize=(12,4.5))
ax[0].hist(dte,80,alpha=.7,label='held-out data rows'); ax[0].hist(tree.query(N[:3000])[0],80,alpha=.7,label='uniform noise rows'); ax[0].legend(); ax[0].set_xlabel('distance to fitted curve'); ax[0].set_yscale('log')
ax[1].bar(np.arange(40),h); ax[1].set_xlabel('position along curve (40 bins)'); ax[1].set_ylabel('rows')
plt.tight_layout(); plt.savefig('plot_fit_quality.png',dpi=100)
