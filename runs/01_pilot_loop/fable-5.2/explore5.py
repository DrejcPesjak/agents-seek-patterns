import numpy as np
from scipy import stats
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import cross_val_predict
X=np.loadtxt('example_X.csv',delimiter=',')
n,d=X.shape
rng=np.random.default_rng(3); R=rng.random((n,d))
# A. can any column be predicted from the others? (circular target -> predict cos & sin)
def pred(X,cols):
    out=[]
    for j in cols:
        O=np.delete(X,j,1)
        F=np.hstack([O,np.cos(2*np.pi*O),np.sin(2*np.pi*O)])
        Y=np.c_[X[:,j],np.cos(2*np.pi*X[:,j]),np.sin(2*np.pi*X[:,j])]
        P=cross_val_predict(ExtraTreesRegressor(60,min_samples_leaf=5,n_jobs=-1,random_state=0),F,Y,cv=4)
        out.append(max(1-((Y[:,k]-P[:,k])**2).sum()/((Y[:,k]-Y[:,k].mean())**2).sum() for k in range(3)))
    return np.array(out)
a=pred(X,range(d)); b=pred(R,range(0,d,5))
print('CV R2 col-from-others: data max %.4f mean %.4f | null max %.4f mean %.4f'%(a.max(),a.mean(),b.max(),b.mean()))
np.save('/tmp/claude-1000/-home-drew99-Projects-agents-and-3d-plots-ex1/4accface-8cd0-43af-9119-b8bc65916e42/scratchpad/r2.npy',a)
# B. real vs fresh-uniform classifier (catches any joint departure)
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
def feats(A): return np.hstack([A,np.cos(2*np.pi*A),np.sin(2*np.pi*A),np.sort(A,1)])
Z=np.vstack([feats(X),feats(rng.random((n,d)))]); y=np.r_[np.ones(n),np.zeros(n)]
print('real-vs-uniform classifier CV acc',cross_val_score(ExtraTreesClassifier(300,min_samples_leaf=3,n_jobs=-1,random_state=0),Z,y,cv=5).mean().round(4),'(0.5 = indistinguishable)')
# C. kernel PCA / spectral on torus embedding
E=np.hstack([np.cos(2*np.pi*X),np.sin(2*np.pi*X)])
for nm,A in (('data',E),('null',np.hstack([np.cos(2*np.pi*R),np.sin(2*np.pi*R)]))):
    s=np.linalg.svd(A-A.mean(0),compute_uv=False); print('torus-embed sv',nm,np.round(s[:3],2),np.round(s[-3:],2))
# D. all-column signed sums mod 1 with random +-1 / small coef vectors won't find a needle; test structured ones
for nm,w in (('all +1',np.ones(d)),('alternating',(-1.)**np.arange(d)),('1..d',np.arange(1,d+1.))):
    v=abs(np.exp(2j*np.pi*(X@w)).mean()); print('global relation',nm,round(v,4),'noise~',round(2/np.sqrt(n),4))
# E. rank-based: Spearman/Kendall max, distance correlation proxy via |x-.5| and (x-.5)^2 features
Fq=np.hstack([np.abs(X-.5),np.abs(X-.25)%0.5]); C=np.corrcoef(np.hstack([X,Fq]).T); np.fill_diagonal(C,0)
for i in range(d): C[i,d+i]=C[d+i,i]=C[i,2*d+i]=C[2*d+i,i]=C[d+i,2*d+i]=C[2*d+i,d+i]=0
Fr=np.hstack([np.abs(R-.5),np.abs(R-.25)%0.5]); C0=np.corrcoef(np.hstack([R,Fr]).T); np.fill_diagonal(C0,0)
for i in range(d): C0[i,d+i]=C0[d+i,i]=C0[i,2*d+i]=C0[2*d+i,i]=C0[d+i,2*d+i]=C0[2*d+i,d+i]=0
print('max corr incl. folded features data',np.abs(C).max().round(4),'null',np.abs(C0).max().round(4))
