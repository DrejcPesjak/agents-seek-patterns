import numpy as np,itertools,time
X=np.load('analysis_outputs/X.npy'); n=1600
f=open('analysis_outputs/log.txt','a',buffering=1)
def log(s):print(s,flush=True);f.write(str(s)+'\n')
# All single-column Fourier harmonics through 16, and pair products through 4.
F=np.exp(2j*np.pi*X[:n,:,None]*np.arange(1,17)).astype(np.complex64).reshape(n,-1)
FK=np.zeros((480,30),int)
for i in range(30):FK[i*16:(i+1)*16,i]=np.arange(1,17)
log('Expanded pair/triple Fourier search: pair coefficients through 4, third coefficient through 16.')
K=[];P=[]
for i,j in itertools.combinations(range(30),2):
 for a,b in itertools.product(range(1,5),range(-4,5)):
  if b==0:continue
  k=np.zeros(30,int);k[i]=a;k[j]=b;K.append(k)
  P.append(F[:,i*16+a-1]*(F[:,j*16+abs(b)-1] if b>0 else F[:,j*16+abs(b)-1].conj()))
K=np.array(K);P=np.array(P).T.copy(); candidates=[]
for conj in [True,False]:
 A=(F.conj().T if conj else F.T)@P/n
 for i,j in zip(*np.where(abs(A)>.11)):
  k=K[j]+(-1 if conj else 1)*FK[i]
  if np.any(k):candidates.append((float(abs(A[i,j])),k))
log('Expanded triple candidates '+str(len(candidates)))
# coefficients through 2 for four-way, includes asymmetric patterns.
mask=np.max(abs(K),axis=1)<=2;P=P[:,mask].copy();K=K[mask]
for start in range(0,len(K),128):
 for conj in [True,False]:
  A=(P[:,start:start+128].conj().T if conj else P[:,start:start+128].T)@P/n
  for i,j in zip(*np.where(abs(A)>.11)):
   k=K[j]+(-1 if conj else 1)*K[start+i]
   if np.any(k):candidates.append((float(abs(A[i,j])),k))
seen=set();out=[]
for v,k in candidates:
 k=k*np.sign(k[np.flatnonzero(k)[0]]);key=tuple(k)
 if key in seen:continue
 seen.add(key);val=abs(np.exp(2j*np.pi*(X[2000:5000]@k)).mean());out.append((val,v,k))
out.sort(key=lambda a:a[0],reverse=True)
for val,v,k in out[:20]:log(f'expanded validation {val:.5f} screen {v:.5f}: {[(i+1,int(a)) for i,a in enumerate(k) if a]}')
log('Expanded search complete, unique candidates '+str(len(out)))
