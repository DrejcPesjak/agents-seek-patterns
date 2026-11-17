# Fourier (character) scan: |mean exp(2*pi*i*(a.x))| for sparse integer vectors a.
# Under independence+uniform marginals, n*|.|^2 ~ Exp(1).
import numpy as np, pandas as pd, itertools, time
X=pd.read_csv('X.csv').values; n,d=X.shape
K=4
E={k:np.exp(2j*np.pi*k*X) for k in range(-K,K+1)}   # n x d each
res=[]
# singles
for k in range(1,K+1):
    s=n*np.abs(E[k].mean(0))**2
    for j in range(d): res.append((s[j],(j+1,),(k,)))
# pairs
for k1 in range(1,K+1):
    for k2 in range(-K,K+1):
        if k2==0: continue
        M=np.abs(E[k1].T@E[k2]/n)**2*n   # d x d: [i,j] = mean e(k1 xi + k2 xj)
        for i in range(d):
            for j in range(d):
                if i<j or (i>j and k1!=abs(k2)) : res.append((M[i,j],(i+1,j+1),(k1,k2)))
res.sort(reverse=True)
print('n tests',len(res),' expected max ~',np.log(len(res)))
for r in res[:15]: print(round(r[0],1),r[1],r[2])
# triples, coeffs up to 2
K3=2; t0=time.time(); res3=[]
for i in range(d):
    for j in range(i+1,d):
        for k1 in range(1,K3+1):
            for k2 in [c for c in range(-K3,K3+1) if c]:
                P=E[k1][:,i]*E[k2][:,j]          # n
                rest=np.arange(j+1,d)
                for k3 in [c for c in range(-K3,K3+1) if c]:
                    s=n*np.abs(P@E[k3][:,rest]/n)**2
                    for t,l in enumerate(rest): res3.append((s[t],(i+1,j+1,l+1),(k1,k2,k3)))
res3.sort(reverse=True)
print('triples: n tests',len(res3),' expected max ~',np.log(len(res3)),time.time()-t0)
for r in res3[:20]: print(round(r[0],1),r[1],r[2])
