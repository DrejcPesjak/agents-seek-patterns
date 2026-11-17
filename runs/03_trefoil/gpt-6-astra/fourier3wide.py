import numpy as np,itertools,time
D=np.load('split.npz');tr=D['train'][:2000];te=D['test'];n,d=tr.shape;K=8
f=open('analysis.log','a')
def log(s):print(s,flush=True);f.write(str(s)+'\n');f.flush()
t=time.time()
labs1=[((j,k),) for j in range(d) for k in range(1,K+1)]
S=np.array([np.exp(2j*np.pi*k*tr[:,j]) for ((j,k),) in labs1],dtype=np.complex64).T.copy()
hits={};pairs=list(itertools.combinations(range(d),2))
for start in range(0,len(pairs),15):
 labs=[((j,k),(l,m)) for j,l in pairs[start:start+15] for k in range(1,K+1) for m in range(-K,K+1) if m]
 P=np.array([np.exp(2j*np.pi*(k*tr[:,j]+m*tr[:,l])) for ((j,k),(l,m)) in labs],dtype=np.complex64).T.copy()
 for sign in [1,-1]:
  C=S.T@(P.conj() if sign==-1 else P)/n
  ii,jj=np.where(abs(C)>.11)
  for i,j in zip(ii,jj):
   dic={}
   for col,k in labs1[i]:dic[col]=dic.get(col,0)+k
   for col,k in labs[j]:dic[col]=dic.get(col,0)+sign*k
   lab=tuple((col,k) for col,k in sorted(dic.items()) if k)
   if not lab:continue
   if lab[0][1]<0:lab=tuple((col,-k) for col,k in lab)
   hits[lab]=float(abs(C[i,j]))
 if start%75==0:log(f'Wide Fourier3 progress {start}/{len(pairs)} elapsed {time.time()-t:.1f}s; candidates {len(hits)}')
ranked=sorted(hits.items(),key=lambda z:-z[1]);log(f'Wide Fourier3 candidates {len(hits)}')
for lab,score in ranked[:100]:
 val=abs(np.exp(2j*np.pi*sum(te[:,col]*k for col,k in lab)).mean())
 log(f'Wide Fourier3 {tuple((col+1,k) for col,k in lab)}: train {score:.5f}, test {val:.5f}')
np.save('fourier3wide_hits.npy',np.array(ranked,dtype=object),allow_pickle=True)
log(f'Wide Fourier3 elapsed {time.time()-t:.1f}s')
