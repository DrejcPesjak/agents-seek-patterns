import numpy as np,itertools,time
D=np.load('split.npz');tr=D['train'][:2000];te=D['test'];n,d=tr.shape
f=open('analysis.log','a')
def log(s):print(s,flush=True);f.write(str(s)+'\n');f.flush()
t=time.time();K=2
labsP=[((j,k),) for j in range(d) for k in range(1,K+1)]
labsP += [((j,k),(l,m)) for j,l in itertools.combinations(range(d),2) for k in range(1,K+1) for m in range(-K,K+1) if m]
P=np.array([np.exp(2j*np.pi*sum(tr[:,col]*k for col,k in lab)) for lab in labsP],dtype=np.complex64).T.copy()
labsT=[tuple(zip(cols,ks)) for cols in itertools.combinations(range(d),3) for ks in itertools.product([1,2],[-2,-1,1,2],[-2,-1,1,2])]
log(f'Fourier5mixed {len(labsT)} triple features by {len(labsP)} pairs, coefficients through 2.')
hits={}
for start in range(0,len(labsT),256):
 labs=labsT[start:start+256]
 T=np.array([np.exp(2j*np.pi*sum(tr[:,col]*k for col,k in lab)) for lab in labs],dtype=np.complex64).T.copy()
 for sign in [1,-1]:
  C=T.T@(P if sign==1 else P.conj())/n
  ii,jj=np.where(abs(C)>.11)
  for i,j in zip(ii,jj):
   dic={}
   for col,k in labs[i]:dic[col]=dic.get(col,0)+k
   for col,k in labsP[j]:dic[col]=dic.get(col,0)+sign*k
   lab=tuple((col,k) for col,k in sorted(dic.items()) if k)
   if not lab:continue
   if lab[0][1]<0:lab=tuple((col,-k) for col,k in lab)
   hits[lab]=max(hits.get(lab,0),float(abs(C[i,j])))
 if start%16384==0:log(f'Fourier5mixed progress {start}/{len(labsT)}, elapsed {time.time()-t:.1f}s, candidates {len(hits)}')
ranked=sorted(hits.items(),key=lambda z:-z[1]);log(f'Fourier5mixed candidates {len(ranked)}')
for lab,score in ranked[:100]:
 val=abs(np.exp(2j*np.pi*sum(te[:,col]*k for col,k in lab)).mean())
 log(f'Fourier5mixed {tuple((col+1,k) for col,k in lab)}: train {score:.5f}, test {val:.5f}')
np.save('fourier5mixed_hits.npy',np.array(ranked,dtype=object),allow_pickle=True)
log(f'Fourier5mixed elapsed {time.time()-t:.1f}s')
