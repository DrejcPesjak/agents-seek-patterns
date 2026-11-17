import numpy as np, pandas as pd
from scipy import stats
X=pd.read_csv('X.csv').values; n,d=X.shape
rng=np.random.default_rng(1)
# (a) rank-transform / grid check
for j in (0,1,2):
    s=np.sort(X[:,j]); print('col',j+1,'first',s[:3],'last',s[-3:],'spacing std/mean',np.diff(s).std()/np.diff(s).mean())
print('repr',[repr(v) for v in X[0,:3]])
# (b) row-level symmetric statistics vs simulation
def feats(Z):
    S=np.sort(Z,1); c=Z-.5
    f={'sum':Z.sum(1),'r2':(c**2).sum(1),'sum_mod1':Z.sum(1)%1,'min':S[:,0],'max':S[:,-1],'med':np.median(Z,1),
       'l1':np.abs(c).sum(1),'l4':(c**4).sum(1),'linf':np.abs(c).max(1),'altsum_mod1':(Z[:,::2].sum(1)-Z[:,1::2].sum(1))%1,
       'prod_log':np.log(Z+1e-300).sum(1),'mingap':np.diff(S,axis=1).min(1),'maxgap':np.diff(S,axis=1).max(1),
       'adjdiff':np.abs(np.diff(Z,axis=1)).sum(1),'adjprod':(c[:,1:]*c[:,:-1]).sum(1),'cos_sum':np.cos(2*np.pi*Z).sum(1),'sin_sum':np.sin(2*np.pi*Z).sum(1)}
    for k in range(d): f[f'os{k}']=S[:,k]
    return f
F=feats(X); R=feats(rng.random((200000,d)))
ps={k:stats.ks_2samp(F[k],R[k]).pvalue for k in F}
print('row-feature KS p-values (smallest 8 of',len(ps),'):',sorted([(round(v,4),k) for k,v in ps.items()])[:8])
# (c) duplicated values across columns / rows (shared values)
allv=X.ravel(); print('unique values overall',len(np.unique(allv)),'of',allv.size)
# decimal digits
print('digits len',pd.read_csv('X.csv',dtype=str).iloc[:5,:3].map(len).values)
# (d) row order: any dependence between consecutive rows? distance to next row vs random row
dn=np.linalg.norm(X[1:]-X[:-1],axis=1); dr=np.linalg.norm(X[rng.permutation(n)]-X,axis=1)
print('consecutive-row dist mean',dn.mean(),'random',dr.mean(), stats.ks_2samp(dn,dr).pvalue)
# trend along the index for each col and Fourier features
t=np.arange(n)/n
print('max |corr(col, index)|',max(abs(np.corrcoef(X[:,j],t)[0,1]) for j in range(d)))
