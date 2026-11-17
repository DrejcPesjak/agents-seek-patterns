import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
X=pd.read_csv('../X.csv').values; n,d=X.shape
rng=np.random.default_rng(1); U=rng.random((n,d))
A=np.vstack([X,U]); y=np.r_[np.ones(n),np.zeros(n)]
for depth,it in [(3,300),(6,300)]:
    clf=HistGradientBoostingClassifier(max_depth=depth,max_iter=it,learning_rate=0.05,max_bins=64)
    p=cross_val_predict(clf,A,y,cv=4,method='predict_proba')[:,1]
    print(depth,'AUC',roc_auc_score(y,p),flush=True)
# pairwise distances distribution: count of close pairs in full space at various scales
# sorted-row features
S=np.sort(A,1)
clf=HistGradientBoostingClassifier(max_depth=4,max_iter=300,learning_rate=0.05)
p=cross_val_predict(clf,S,y,cv=4,method='predict_proba')[:,1]; print('sorted AUC',roc_auc_score(y,p))
