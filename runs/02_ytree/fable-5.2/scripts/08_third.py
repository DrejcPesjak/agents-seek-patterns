# Is there a third relation? (a) 5-subset torus-kNN scan excluding supersets of the two known quads,
# (b) does the position on the Y (arm, distance) depend on characters of any columns?
import numpy as np, pandas as pd, itertools
from scipy.spatial import cKDTree
from multiprocessing import Pool
X = pd.read_csv('X.csv').values
n,d=X.shape; x=lambda k:X[:,k-1]
QA={9,22,24,29}; QB={0,5,10,23}
def knn(cols):
    t=cKDTree(X[:,cols],boxsize=1.0); dd,_=t.query(X[:,cols],k=2); return np.log(dd[:,1]).mean()
if __name__=='__main__':
    u=(2*x(10)+2*x(23)+x(25)+x(30))%1; v=(2*x(1)-2*x(6)-x(11)-x(24))%1
    c=np.load('scripts/yfit.npy')[0]
    du,dv=u-c[0],v-c[1]; rad=np.hypot(du,dv); ang=np.arctan2(dv,du)
    feats={'du':du,'dv':dv,'rad':rad,'cos3ang':np.cos(3*ang)*(rad>.02),'arm0':(abs(ang-0.33)<.6)*1.,'arm1':(abs(ang-2.48)<.6)*1.}
    E={k:np.exp(2j*np.pi*k*X) for k in (-2,-1,1,2)}
    for nm,f in feats.items():
        f=(f-f.mean())/f.std(); best=[]
        for j in range(d):
            for k in (1,2): best.append((abs((f*E[k][:,j]).mean()),(j+1,k)))
        for i,j in itertools.combinations(range(d),2):
            for a in (1,2):
                for b in (-2,-1,1,2): best.append((abs((f*E[a][:,i]*E[b][:,j]).mean()),(i+1,a,j+1,b)))
        best.sort(reverse=True); print(nm,'null max ~',round(np.sqrt(np.log(len(best))/n)*1.3,3),[(round(float(r),3),t) for r,t in best[:4]])
    combos=[list(cb) for cb in itertools.combinations(range(d),5) if not (QA<=set(cb) or QB<=set(cb))]
    print('5-subsets to scan',len(combos))
    with Pool(15) as p: r=np.array(p.map(knn,combos,chunksize=200))
    z=(r-np.median(r))/r.std(); o=np.argsort(z)[:8]
    print('knn m=5 mean',r.mean(),'sd',r.std(),'lowest z:',[(tuple(int(q)+1 for q in combos[i]),round(float(z[i]),2)) for i in o],'max z',z.max().round(2))
