import numpy as np, pandas as pd, itertools
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
X=pd.read_csv('X.csv').values; n,d=X.shape
C=[3,6,8,10,29]; Z=X[:,[c-1 for c in C]]
u=(Z@np.array([1,-1,2,-2,-1]))%1
ang=np.angle(np.exp(2j*np.pi*u).mean())/(2*np.pi)%1
r=(u-ang+.5)%1-.5
print('center',ang,'resid std',r.std(),'quantiles',np.quantile(r,[.001,.01,.25,.5,.75,.99,.999]).round(4))
print('frac |r|>0.1:',(abs(r)>0.1).mean())
# all characters on the 5 columns, coefs -4..4
K=4; res=[]
for co in itertools.product(range(-K,K+1),repeat=5):
    co=np.array(co)
    if not co.any() or co[np.nonzero(co)[0][0]]<0: continue
    s=n*abs(np.exp(2j*np.pi*(Z@co)).mean())**2
    if s>18: res.append((s,dict(zip(C,co.tolist()))))
for s,c in sorted(res,key=lambda t:-t[0])[:30]: print(round(s,1),{k:v for k,v in c.items() if v})
# does residual depend on anything?  correlate r with sin/cos of all columns and of sub-combos
best=[]
for j in range(d):
    for k in (1,2):
        for f in (np.sin,np.cos):
            best.append((abs(np.corrcoef(r,f(2*np.pi*k*X[:,j]))[0,1]),j+1,k,f.__name__))
print('resid vs columns top:',sorted(best,reverse=True)[:5])
fig,ax=plt.subplots(1,3,figsize=(15,4.5))
ax[0].hist(u,200); ax[0].set_title('u = mod(x3 - x6 + 2*x8 - 2*x10 - x29, 1)')
ax[1].hist(r,200); ax[1].set_title('residual around centre'); ax[1].set_yscale('log')
ax[2].plot(r,'.',ms=1); ax[2].set_title('residual vs row index')
plt.tight_layout(); plt.savefig('plot_relation.png',dpi=110)
