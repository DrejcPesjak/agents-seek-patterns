import numpy as np, pandas as pd, itertools
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
X = pd.read_csv('X.csv').values
n=len(X); x=lambda k:X[:,k-1]
u=(2*x(10)+2*x(23)+x(25)+x(30))%1
v=(2*x(1)-2*x(6)-x(11)-x(24))%1
def circ(t):
    m=np.exp(2j*np.pi*t).mean(); mu=(np.angle(m)/2/np.pi)%1
    r=((t-mu+.5)%1)-.5
    return mu,r
for nm,t in (('u',u),('v',v)):
    mu,r=circ(t); print(nm,'circ mean',mu,'resid sd',r.std(),'quantiles',np.quantile(r,[0,.001,.01,.25,.5,.75,.99,.999,1]).round(4))
    print('  hist20',np.histogram(t,20,(0,1))[0])
mu_u,ru=circ(u); mu_v,rv=circ(v)
print('corr(ru,rv)',np.corrcoef(ru,rv)[0,1],'corr abs',np.corrcoef(abs(ru),abs(rv))[0,1])
# do residuals depend on any column (any harmonic)?
for nm,r in (('ru',ru),('rv',rv)):
    best=[]
    for j in range(30):
        for k in (1,2,3,4):
            c=abs(np.mean((r-r.mean())/r.std()*np.exp(2j*np.pi*k*X[:,j])))
            best.append((round(float(c),3),j+1,k))
    best.sort(reverse=True); print(nm,'top dependence on column harmonics (null ~0.03):',best[:6])
    c=[round(float(np.corrcoef(abs(r),np.abs(X[:,j]-.5))[0,1]),3) for j in range(30)]; print('  |r| vs |x-.5| max',max(np.abs(c)))
fig,ax=plt.subplots(1,3,figsize=(15,4.5))
ax[0].hist(u,100); ax[0].set_title('u = (2x10+2x23+x25+x30) mod 1')
ax[1].hist(v,100); ax[1].set_title('v = (2x1-2x6-x11-x24) mod 1')
ax[2].scatter(u,v,s=2,alpha=.4); ax[2].set_xlim(0,1); ax[2].set_ylim(0,1); ax[2].set_xlabel('u'); ax[2].set_ylabel('v'); ax[2].set_title('u vs v')
plt.tight_layout(); plt.savefig('plots/05_uv.png',dpi=110)
