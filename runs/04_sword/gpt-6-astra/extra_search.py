import numpy as np,itertools
from scipy import special
X=np.load('analysis_outputs/X.npy');n=2000;d=30
f=open('analysis_outputs/log.txt','a',buffering=1)
def log(s):print(s,flush=True);f.write(str(s)+'\n')
# Nonlinear pair features: Fourier transformed products, differences, ratios, radial quantities.
# Standardize by analytic/independently simulated null; compare only disjoint variables.
rng=np.random.default_rng(201);U=rng.random((12000,d))
features=[];desc=[];inds=[];null=[]
for i,j in itertools.combinations(range(d),2):
 a,b=X[:n,i],X[:n,j];u,v=U[:,i],U[:,j]
 funcs=[('product',lambda a,b:a*b),('square_sum',lambda a,b:a*a+b*b),('square_diff',lambda a,b:a*a-b*b),('ratio',lambda a,b:a/(b+.001)),('ratio_reverse',lambda a,b:b/(a+.001)),('radius',lambda a,b:np.sqrt(a*a+b*b)),('angle',lambda a,b:np.arctan2(a-.5,b-.5)/(2*np.pi))]
 for name,func in funcs:
  for h in [1,2]:
   z=np.exp(2j*np.pi*h*func(a,b));z0=np.exp(2j*np.pi*h*func(u,v));mu=z0.mean();sd=np.sqrt(1-abs(mu)**2)
   features.append((z-mu)/sd);null.append((mu,sd));desc.append((name,h,i+1,j+1));inds.append({i,j})
P=np.array(features,dtype=np.complex64).T.copy();candidates=[]
for start in range(0,len(desc),128):
 for conj in [True,False]:
  A=(P[:,start:start+128].conj().T if conj else P[:,start:start+128].T)@P/n
  for i,j in zip(*np.where(abs(A)>.17)):
   if inds[start+i].isdisjoint(inds[j]): candidates.append((float(abs(A[i,j])),start+i,j,conj))
log('Nonlinear disjoint pair feature screen candidates '+str(len(candidates)))
for v,i,j,conj in sorted(candidates,reverse=True)[:30]:log(str((v,desc[i],desc[j],conj)))
# GF2 rank and parity up to six columns for binary digits at several resolutions.
for scale in [1,2,4,8,16,32,64,128]:
 B=(np.floor(X*2*scale).astype(int)%2).astype(np.uint8);A=B.copy();rank=0
 for j in range(d):
  w=np.flatnonzero(A[rank:,j])
  if len(w):
   k=rank+w[0];A[[rank,k]]=A[[k,rank]];m=np.flatnonzero(A[:,j]);m=m[m!=rank];A[m]^=A[rank];rank+=1
 log(f'Binary-digit GF2 rank scale {scale}: {rank}')
 S=1-2*B[:2000].astype(np.float32);K=list(itertools.combinations(range(d),3));P=np.array([S[:,i]*S[:,j]*S[:,k] for i,j,k in K]).T.copy()
 best=[]
 for start in range(0,len(K),128):
  A=P[:,start:start+128].T@P/2000
  for i,j in zip(*np.where((abs(A)>.14)&(abs(A)<.9999))):
   kk=set(K[start+i])^set(K[j]);val=abs(np.prod(1-2*B[3000:6000,sorted(kk)].astype(float),axis=1).mean())
   best.append((val,float(A[i,j]),sorted(x+1 for x in kk)))
 if best:log('Bit parity candidates '+str(sorted(best,reverse=True)[:3]))
