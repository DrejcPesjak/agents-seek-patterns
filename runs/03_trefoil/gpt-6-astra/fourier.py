import numpy as np,itertools,time
from analyze import log
A=np.load('data.npy');n,d=A.shape
# Use disjoint train and test halves: candidate search never scores on test.
rng=np.random.default_rng(321);ix=rng.permutation(n);tr=A[ix[:4000]];te=A[ix[4000:]]
np.savez('split.npz',train=tr,test=te)
t=time.time()
F=np.exp(2j*np.pi*tr[:,:,None]*np.arange(1,5)).reshape(len(tr),-1)
labels=[((j+1,k),) for j in range(d) for k in range(1,5)]
# pair products of individual +- harmonics, and single harmonics; Fourier support up to 4.
features=[];labs=[]
for j in range(d):
 for k in [1,2]:
  features.append(np.exp(2j*np.pi*k*tr[:,j]));labs.append(((j+1,k),))
for j,l in itertools.combinations(range(d),2):
 for k in [1,2]:
  for m in [-2,-1,1,2]:
   features.append(np.exp(2j*np.pi*(k*tr[:,j]+m*tr[:,l])));labs.append(((j+1,k),(l+1,m)))
F=np.array(features,dtype=np.complex64).T.copy();p=F.shape[1]
log(f'Sparse Fourier search: {p} features, coefficients 1 or 2; correlations discover up to four participating columns.')
hits={}
for start in range(0,p,256):
 C=F[:,start:start+256].T@F.conj()/len(tr)
 for ii,row in enumerate(C):
  i=start+ii
  for j in np.flatnonzero(abs(row)>.065):
   if j>=i:continue
   dic={}
   for col,k in labs[i]:dic[col]=dic.get(col,0)+k
   for col,k in labs[j]:dic[col]=dic.get(col,0)-k
   lab=tuple((col,k) for col,k in sorted(dic.items()) if k)
   if not lab:continue
   if lab[0][1]<0:lab=tuple((col,-k) for col,k in lab)
   hits[lab]=max(hits.get(lab,0),float(abs(row[j])))
 if start%1024==0: print(start,p,'seconds',time.time()-t,flush=True)
log(f'Fourier candidate count {len(hits)}')
ranked=sorted(hits.items(),key=lambda z:-z[1])
for lab,score in ranked[:100]:
 val=abs(np.exp(2j*np.pi*sum(te[:,col-1]*k for col,k in lab)).mean())
 log(f'Fourier {lab}: train {score:.5f}, test {val:.5f}')
np.save('fourier_hits.npy',np.array(ranked,dtype=object),allow_pickle=True)
log('Fourier search seconds '+str(time.time()-t))
