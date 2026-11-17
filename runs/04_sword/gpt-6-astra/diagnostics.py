import numpy as np
from scipy import stats,special
from sklearn.ensemble import ExtraTreesClassifier,RandomForestRegressor,HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
X=np.load('analysis_outputs/X.npy');rng=np.random.default_rng(7364)
f=open('analysis_outputs/log.txt','a',buffering=1)
def log(s): print(s,flush=True);f.write(str(s)+'\n')
log('Gaussian-score covariance eigenvalues '+str(np.linalg.eigvalsh(np.cov(special.ndtri(X).T))))
log('Row sums against sum-of-uniform simulation: '+str(stats.ks_2samp(X.sum(1),rng.random(X.shape).sum(1))))
log('Row squared radius: '+str(stats.ks_2samp(((X-.5)**2).sum(1),((rng.random(X.shape)-.5)**2).sum(1))))
log('Exact duplicate rows '+str(len(X)-len(np.unique(X,axis=0))))
log('Maximum absolute lag-1 correlation '+str(max(abs(np.corrcoef(v[:-1],v[1:])[0,1]) for v in X.T)))
for circular in [False,True]:
 Y=np.c_[np.sin(2*np.pi*X),np.cos(2*np.pi*X)] if circular else X
 U=rng.random(X.shape); U=np.c_[np.sin(2*np.pi*U),np.cos(2*np.pi*U)] if circular else U
 a=NearestNeighbors(n_neighbors=2,n_jobs=4).fit(Y).kneighbors(Y)[0][:,1]
 b=NearestNeighbors(n_neighbors=2,n_jobs=4).fit(U).kneighbors(U)[0][:,1]
 log(f'Nearest-neighbor distances circular={circular}, real/null mean {a.mean():.5f}/{b.mean():.5f}, KS {stats.ks_2samp(a,b)}')
 fig,ax=plt.subplots();ax.hist(a,bins=50,alpha=.5,density=True,label='X.csv');ax.hist(b,bins=50,alpha=.5,density=True,label='independent uniform');ax.legend();ax.set_title(f'Nearest-neighbor distances; circular={circular}');fig.savefig(f'analysis_outputs/neighbors_{circular}.png');plt.close(fig)
tr=np.r_[X[:4000],rng.random((4000,30))];te=np.r_[X[4000:],rng.random((4000,30))];y=np.r_[np.ones(4000),np.zeros(4000)]
for clf in [ExtraTreesClassifier(n_estimators=400,min_samples_leaf=3,n_jobs=4,random_state=7),HistGradientBoostingClassifier(max_iter=300,max_leaf_nodes=15,l2_regularization=5,random_state=7)]:
 clf.fit(tr,y);pred=clf.predict_proba(te)[:,1];log(f'Independent holdout real-vs-uniform classifier {type(clf).__name__}: AUC {roc_auc_score(y,pred):.5f}')
