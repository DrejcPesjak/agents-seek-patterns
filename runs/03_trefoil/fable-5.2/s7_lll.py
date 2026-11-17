# Dense integer-relation test: is there an integer vector a with a.x ≈ const (mod 1)?
# Lattice: rows (A*e_i | Wint*y_i) and (0 | Wint*e_r), y = differences of data rows. Short vector => relation.
import numpy as np, pandas as pd, sys, time
from fpylll import IntegerMatrix, LLL, BKZ
def test(X,R=90,W=100,seed=0,bkz=20):
    n,d=X.shape; rng=np.random.default_rng(seed)
    idx=rng.choice(n,2*R,replace=False); Y=X[idx[:R]]-X[idx[R:]]
    Wint=10**9; A=int(Wint/W)
    M=IntegerMatrix(d+R,d+R)
    for i in range(d):
        M[i,i]=A
        for r in range(R): M[i,d+r]=int(round(Wint*Y[r,i]))
    for r in range(R): M[d+r,d+r]=Wint
    LLL.reduction(M)
    if bkz: BKZ.reduction(M,BKZ.Param(block_size=bkz))
    out=[]
    for row in range(3):
        v=np.array([M[row,c] for c in range(d+R)],dtype=float)
        a=np.round(v[:d]/A).astype(int)
        out.append((np.linalg.norm(v)/A,a))
    return out
def check(X,a):
    z=np.exp(2j*np.pi*(X@a)).mean(); return len(X)*abs(z)**2
if __name__=='__main__':
    X=pd.read_csv('X.csv').values; n,d=X.shape
    rng=np.random.default_rng(5)
    # positive control: planted dense relation, 12 columns, noise sigma
    for sig in (0.003,0.02):
        Z=rng.random((n,d)); S=rng.choice(d,12,replace=False)
        Z[:,S[0]]=(0.3-Z[:,S[1:]].sum(1)+sig*rng.standard_normal(n))%1
        for W in (10,100):
            t0=time.time(); o=test(Z,W=W)
            print(f'CONTROL sig={sig} W={W}: norm {o[0][0]:.1f}, |a|1={abs(o[0][1]).sum()}, stat={check(Z,o[0][1]):.0f} (n={n} if perfect)  [{time.time()-t0:.0f}s]')
    for W in (10,100,1000):
        for seed in (0,1):
            o=test(X,W=W,seed=seed)
            print(f'DATA W={W} seed={seed}:',[(round(nm,1),int(abs(a).sum()),round(check(X,a),1)) for nm,a in o])
