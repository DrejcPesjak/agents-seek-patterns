import numpy as np,time,itertools
X=np.load('working_data.npz')['X'];rng=np.random.default_rng(2026);order=rng.permutation(len(X));A=X[order[:2000]];B=X[order[2000:]]
def log(s):
 print(s,flush=True)
 with open('analysis.log','a') as f:f.write(s+'\n')
# Every sparse Fourier frequency involving <= 4 columns, coefficients +/-1 or +/-2.
vec=[]
for i in range(30):
 for a in (1,2):
  v=np.zeros(30,dtype=np.int8);v[i]=a;vec.append(v)
for i in range(30):
 for j in range(i):
  for a,b in itertools.product((1,2),(-2,-1,1,2)):
   v=np.zeros(30,dtype=np.int8);v[i]=a;v[j]=b;vec.append(v)
V=np.array(vec);Z=np.exp(2j*np.pi*(A@V.T)).astype(np.complex64);n=len(V)
log(f'Fourier search: {n} one/two-column frequencies; testing sums/differences to reach four-column relations on 2000 discovery rows.')
best=[]
for start in range(0,n,160):
 z=Z[:,start:start+160]
 for sign in (1,-1):
  C=(z.T@(Z if sign==1 else Z.conj()))/len(A)
  S=np.abs(C)
  # eliminate identically zero frequencies
  if sign==-1:
   for k in range(len(z.T)):S[k,start+k]=0
  inds=np.argpartition(S.ravel(),-40)[-40:]
  for idx in inds:
   a,b=np.unravel_index(idx,S.shape);v=V[start+a].astype(int)+sign*V[b]
   nz=np.flatnonzero(v)
   if not len(nz):continue
   if v[nz[0]]<0:v=-v
   best.append((float(S[a,b]),tuple(v)))
 if start%640==0:log(f'Fourier progress {start}/{n}; best magnitude {max(s for s,v in best):.4f}')
D={}
for s,v in best:D[v]=max(s,D.get(v,0))
top=sorted([(s,v) for v,s in D.items()],reverse=True)[:100]
results=[]
for s,v in top:
 v=np.array(v);z=np.exp(2j*np.pi*(B@v));t=abs(z.mean());results.append((s,t,v.tolist()))
results.sort(key=lambda a:a[1],reverse=True)
log('Top independent-validation Fourier relations: '+str(results[:30]))
np.savez('fourier_results.npz',results=np.array([r[2] for r in results]),train=[r[0] for r in results],test=[r[1] for r in results])
