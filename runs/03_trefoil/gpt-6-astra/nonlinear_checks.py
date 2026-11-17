import numpy as np,time,itertools
from scipy.stats import kstest
from scipy.spatial.distance import pdist
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
A=np.load('data.npy');D=np.load('split.npz');tr=D['train'];te=D['test'];rng=np.random.default_rng(701)
f=open('analysis.log','a')
def log(s):print(s,flush=True);f.write(str(s)+'\n');f.flush()
log('Marginal uniform KS p-values '+str([round(kstest(A[:,j],'uniform').pvalue,5) for j in range(A.shape[1])]))
mi=[]
for j in range(A.shape[1]):
 vals=mutual_info_regression(np.delete(A,j,axis=1),A[:,j],n_neighbors=5,random_state=1)
 for k,v in zip([k for k in range(A.shape[1]) if k!=j],vals):
  if k>j:mi.append((v,j+1,k+1))
log('Largest pairwise kNN mutual information '+str(sorted(mi,reverse=True)[:15]))
# Row order may matter; autocorrelation checks.
log('Largest abs lag1 per-column autocorrelation '+str(max(abs(np.corrcoef(A[:-1,j],A[1:,j])[0,1]) for j in range(A.shape[1]))))
for name,model in [('ExtraTrees',ExtraTreesClassifier(n_estimators=350,min_samples_leaf=8,n_jobs=4,random_state=1)),('HistGradientBoosting',HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=15,l2_regularization=10,random_state=1))]:
 tn=np.column_stack([rng.permutation(tr[:,j]) for j in range(tr.shape[1])]);en=np.column_stack([rng.permutation(te[:,j]) for j in range(te.shape[1])])
 model.fit(np.r_[tr,tn],np.r_[np.ones(len(tr)),np.zeros(len(tr))]);pred=model.predict_proba(np.r_[te,en])[:,1]
 log(f'{name}: heldout original vs independently shuffled columns AUC {roc_auc_score(np.r_[np.ones(len(te)),np.zeros(len(te))],pred):.5f}')
# Raw and periodic nearest-neighbor geometry relative to null.
from scipy.spatial import cKDTree
for name,coords,null in [('ordinary',A,rng.uniform(size=A.shape)),('torus',A,rng.uniform(size=A.shape))]:
 if name=='torus': t=cKDTree(coords,boxsize=1);nt=cKDTree(null,boxsize=1)
 else:t=cKDTree(coords);nt=cKDTree(null)
 dd=t.query(coords[:1000],k=11,workers=4)[0];nd=nt.query(null[:1000],k=11,workers=4)[0]
 log(f'{name} nearest-neighbor distances data {np.quantile(dd[:,1],[.1,.5,.9])}, null {np.quantile(nd[:,1],[.1,.5,.9])}; intrinsic dimension estimate data {np.mean(9/np.log(dd[:,-1,None]/dd[:,1:10]).sum(1)):.3f}, null {np.mean(9/np.log(nd[:,-1,None]/nd[:,1:10]).sum(1)):.3f}')
