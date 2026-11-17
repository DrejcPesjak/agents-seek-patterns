import numpy as np,warnings
from scipy import stats,special
from sklearn.decomposition import FastICA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
X=np.load('analysis_outputs/X.npy');Y=special.ndtri(X);f=open('analysis_outputs/log.txt','a',buffering=1)
def log(s): print(s,flush=True);f.write(str(s)+'\n')
for fun in ['logcosh','exp','cube']:
 with warnings.catch_warnings():
  warnings.simplefilter('ignore');model=FastICA(n_components=30,fun=fun,max_iter=800,tol=.001,random_state=4,whiten='unit-variance').fit(Y[:3000])
 tr=model.transform(Y[:3000]);a=model.transform(Y[5000:]);b=model.transform(Y[3000:5000]);p=np.array([stats.normaltest(a[:,j]).pvalue for j in range(30)]);j=p.argmin();w=model.components_[j];w=w/np.linalg.norm(w)
 log(f'ICA follow-up {fun}: selected component {j}, initial p {p[j]:.6g}, independent middle rows p {stats.normaltest(b[:,j]).pvalue:.6g}; initial skew/kurtosis {stats.skew(a[:,j]):.4f}/{stats.kurtosis(a[:,j]):.4f}; middle {stats.skew(b[:,j]):.4f}/{stats.kurtosis(b[:,j]):.4f}; weights '+str(w))
 if fun=='exp':
  fig,axs=plt.subplots(1,2,figsize=(10,4));stats.probplot(a[:,j],plot=axs[0]);stats.probplot(b[:,j],plot=axs[1]);axs[0].set_title('Selected ICA component: initial test');axs[1].set_title('Selected ICA component: independent check');fig.tight_layout();fig.savefig('analysis_outputs/ica_check.png')
