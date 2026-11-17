# What does the residual r of the relation depend on? Scan characters exp(2πi b.x) against r, r^2.
import numpy as np, pandas as pd, itertools
X=pd.read_csv('X.csv').values; n,d=X.shape
a=np.zeros(d); a[[2,5,7,9,28]]=[1,-1,2,-2,-1]
r=((X@a)-0.5258+.5)%1-.5
T={'r':r-r.mean(),'r2':r**2-(r**2).mean(),'r3':r**3-(r**3).mean()}
T={k:(v/v.std()).astype(np.complex64) for k,v in T.items()}
def scan(r_,K,name):
    kk=[k for k in range(-K,K+1) if k]; out=[]
    for idx in itertools.combinations(range(d),r_):
        cos=np.array([c for c in itertools.product(kk,repeat=r_) if c[0]>0],dtype=float)
        E=np.exp(2j*np.pi*(X[:,idx]@cos.T)).astype(np.complex64)
        for tn,t in T.items():
            s=np.abs(t@E)**2/n     # ~Exp(1)*... under null (2x for real target ~ Exp(1/2) scale)
            j=s.argmax(); out.append((float(s[j]),tn,{idx[q]+1:int(cos[j][q]) for q in range(r_)}))
    out.sort(key=lambda t:-t[0]); print(name,[ (round(o[0],1),o[1],o[2]) for o in out[:6]])
scan(1,6,'singles'); scan(2,3,'pairs'); scan(3,2,'triples'); scan(4,1,'quads')
