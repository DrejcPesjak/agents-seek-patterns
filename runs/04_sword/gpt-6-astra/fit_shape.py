import numpy as np
from scipy.spatial.distance import cdist
from scipy.interpolate import splprep,splev
from scipy.sparse import coo_matrix,eye,diags
from scipy.sparse.linalg import spsolve
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
Y=np.load('analysis_outputs/decoded.npy');tr=Y[:5000];te=Y[5000:];rng=np.random.default_rng(16)
f=open('analysis_outputs/log.txt','a',buffering=1)
def log(s):print(s,flush=True);f.write(str(s)+'\n')
# Find the closed outline without filling its interior.
C=KMeans(n_clusters=90,n_init=10,random_state=17).fit(tr).cluster_centers_;D=cdist(C,C)
def twoopt(order):
 order=order.copy();n=len(order)
 for repeat in range(100):
  improved=False
  for i in range(n-2):
   for j in range(i+2,n if i else n-1):
    a,b,c,d=order[i],order[(i+1)%n],order[j],order[(j+1)%n]
    if D[a,c]+D[b,d]<D[a,b]+D[c,d]-1e-10:
     order[i+1:j+1]=order[i+1:j+1][::-1];improved=True
  if not improved:break
 return order
best=None
for start in range(20):
 order=[start];unused=set(range(len(C)))-{start}
 while unused:
  j=min(unused,key=lambda j:D[order[-1],j]);order.append(j);unused.remove(j)
 order=twoopt(np.array(order));length=D[order,np.roll(order,-1)].sum()
 if best is None or length<best[0]:best=(length,order)
C=C[best[1]]
def resample(P,m):
 Q=np.r_[P,P[:1]];l=np.r_[0,np.cumsum(np.linalg.norm(np.diff(Q,axis=0),axis=1))];t=np.linspace(0,l[-1],m,endpoint=False)
 return np.c_[np.interp(t,l,Q[:,0]),np.interp(t,l,Q[:,1])]
def project(X,P,chunk=2000):
 V=np.roll(P,-1,axis=0)-P;vv=(V*V).sum(1);ii=[];tt=[];dd=[]
 for start in range(0,len(X),chunk):
  A=X[start:start+chunk,None,:]-P[None,:,:];t=np.clip((A*V[None,:,:]).sum(2)/vv[None,:],0,1);d=((A-t[:,:,None]*V[None,:,:])**2).sum(2);i=d.argmin(1);ii.extend(i);tt.extend(t[np.arange(len(i)),i]);dd.extend(np.sqrt(d[np.arange(len(i)),i]))
 return np.array(ii),np.array(tt),np.array(dd)
# Elastic principal curve: estimate a denoised centerline, regularize its second differences.
P=resample(C,160);m=len(P);E=np.eye(m);L=np.roll(E,1,axis=0)+np.roll(E,-1,axis=0)-2*E;L=np.zeros((m,m))
for i in range(m):L[i,i]=-2;L[i,(i-1)%m]=1;L[i,(i+1)%m]=1
reg=8
for it in range(80):
 i,t,d=project(tr,P);rows=np.repeat(np.arange(len(tr)),2);cols=np.c_[i,(i+1)%m].ravel();vals=np.c_[1-t,t].ravel();A=coo_matrix((vals,(rows,cols)),shape=(len(tr),m)).tocsr()
 i,t,d=project(tr,P)
 rows=np.repeat(np.arange(len(tr)),2);cols=np.c_[i,(i+1)%m].ravel();vals=np.c_[1-t,t].ravel();A=coo_matrix((vals,(rows,cols)),shape=(len(tr),m)).toarray()
 B=A.T@A+reg*(L.T@L)+1e-8*np.eye(m);Q=np.linalg.solve(B,A.T@tr)
 P=resample(Q,m)
 if it%20==0:log(f'Principal curve iteration {it}: train median distance {np.median(d):.6f}')
np.save('analysis_outputs/curve.npy',P)
for name,X in [('train',tr),('holdout',te),('uniform noise',rng.random((100000,2)))]:
 _,_,d=project(X,P);log(f'Distance to sword outline {name}: median={np.median(d):.6f}, q90={np.quantile(d,.9):.6f}, q95={np.quantile(d,.95):.6f}, q99={np.quantile(d,.99):.6f}; fraction within .01={np.mean(d<.01):.6f}')
fig,axs=plt.subplots(1,3,figsize=(16,6))
for ax,X,title in zip(axs,[tr,te,rng.random((8000,2))],['Fit rows (5,000)','Held-out rows (3,000)','Independent uniform noise']):
 ax.scatter(X[:,0],X[:,1],s=2,alpha=.25,label='mapped rows');Q=np.r_[P,P[:1]];ax.plot(Q[:,0],Q[:,1],color='crimson',lw=1.5,label='drawing');ax.set(aspect='equal',xlim=(.25,.7),ylim=(.2,.88),title=title,xlabel='u',ylabel='v');ax.legend(loc='upper right',fontsize=8)
fig.tight_layout();fig.savefig('analysis_outputs/sword_validation.png',dpi=180)
# Zoomed comparison
fig,ax=plt.subplots(figsize=(7,10));ax.scatter(te[:,0],te[:,1],s=5,alpha=.3);Q=np.r_[P,P[:1]];ax.plot(Q[:,0],Q[:,1],color='crimson',lw=1);ax.set(aspect='equal',title='Recovered sword outline: independent held-out rows',xlabel='u',ylabel='v');fig.tight_layout();fig.savefig('analysis_outputs/sword_detail.png',dpi=180)
