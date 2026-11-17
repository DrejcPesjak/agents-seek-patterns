import numpy as np, json
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
P=np.load('uv.npy')
C=KMeans(70,n_init=5,random_state=0).fit(P).cluster_centers_
# TSP: nearest neighbour + 2-opt
def tourlen(o): return np.linalg.norm(C[o]-C[np.roll(o,-1)],axis=1).sum()
o=[0];rest=set(range(1,len(C)))
while rest:
    j=min(rest,key=lambda k:np.linalg.norm(C[k]-C[o[-1]]));o.append(j);rest.remove(j)
o=np.array(o);imp=True
while imp:
    imp=False
    for i in range(1,len(o)-1):
        for j in range(i+1,len(o)):
            new=np.r_[o[:i],o[i:j+1][::-1],o[j+1:]]
            if tourlen(new)<tourlen(o)-1e-12:o=new;imp=True
poly=C[o]; print('tour len',tourlen(o))
def project(P,poly):
    # arc-length param of nearest point on closed polyline
    A=poly;B=np.roll(poly,-1,0);L=np.linalg.norm(B-A,axis=1);cum=np.r_[0,np.cumsum(L)]
    best=np.full(len(P),np.inf);s=np.zeros(len(P))
    for i in range(len(A)):
        t=np.clip(((P-A[i])@(B[i]-A[i]))/L[i]**2,0,1);q=A[i]+t[:,None]*(B[i]-A[i]);dd=np.linalg.norm(P-q,axis=1)
        m=dd<best;best[m]=dd[m];s[m]=cum[i]+t[m]*L[i]
    return s,best,cum[-1]
for it in range(6):
    s,dist,T=project(P,poly)
    M=180;centers=(np.arange(M)+.5)*T/M;h=T/M*1.2
    new=[]
    for c in centers:
        dd=np.abs((s-c+T/2)%T-T/2);w=np.exp(-.5*(dd/h)**2);new.append((P*w[:,None]).sum(0)/w.sum())
    poly=np.array(new)
s,dist,T=project(P,poly)
print('curve length',T,'residual dist: mean %.4f  rms %.4f  p99 %.4f max %.4f'%(dist.mean(),np.sqrt((dist**2).mean()),np.percentile(dist,99),dist.max()))
print('density along curve (20 bins)',np.histogram(s,20,range=(0,T))[0])
np.save('poly.npy',poly)
fig,ax=plt.subplots(figsize=(9,9));ax.scatter(*P.T,s=2,alpha=.4);ax.plot(*np.vstack([poly,poly[:1]]).T,'r.-',lw=1,ms=3);ax.set_aspect('equal');ax.set_title('fitted closed curve (180 pts) over data');ax.set_xlabel('u');ax.set_ylabel('v')
plt.savefig('p_fit.png',dpi=80)
