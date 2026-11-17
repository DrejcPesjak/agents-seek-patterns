import numpy as np,itertools,time
D=np.load('split.npz');tr=D['train'][:2000];te=D['test'];n,d=tr.shape;K=5
f=open('analysis.log','a')
def log(s):print(s,flush=True);f.write(str(s)+'\n');f.flush()
t=time.time();labs=[((j,k),) for j in range(d) for k in range(1,K+1)]
labs += [((j,k),(l,m)) for j,l in itertools.combinations(range(d),2) for k in range(1,K+1) for m in range(-K,K+1) if m]
F=np.array([np.exp(2j*np.pi*sum(tr[:,col]*k for col,k in lab)) for lab in labs],dtype=np.complex64).T.copy();p=F.shape[1]
log(f'Wide Fourier4: {p} features, coefficients through {K}.')
hits={}
for start in range(0,p,192):
 C=F[:,start:start+192].T@F.conj()/n
 for ii,row in enumerate(C):
  i=start+ii
  for j in np.flatnonzero(abs(row)>.11):
   if j>=i:continue
   dic={}
   for col,k in labs[i]:dic[col]=dic.get(col,0)+k
   for col,k in labs[j]:dic[col]=dic.get(col,0)-k
   lab=tuple((col,k) for col,k in sorted(dic.items()) if k)
   if not lab:continue
   if lab[0][1]<0:lab=tuple((col,-k) for col,k in lab)
   hits[lab]=max(hits.get(lab,0),float(abs(row[j])))
 if start%3072==0:log(f'Wide Fourier4 progress {start}/{p}, elapsed {time.time()-t:.1f}s, candidates {len(hits)}')
ranked=sorted(hits.items(),key=lambda z:-z[1]);log(f'Wide Fourier4 candidates {len(ranked)}')
for lab,score in ranked[:100]:
 val=abs(np.exp(2j*np.pi*sum(te[:,col]*k for col,k in lab)).mean())
 log(f'Wide Fourier4 {tuple((col+1,k) for col,k in lab)}: train {score:.5f}, test {val:.5f}')
np.save('fourier4wide_hits.npy',np.array(ranked,dtype=object),allow_pickle=True)
log(f'Wide Fourier4 elapsed {time.time()-t:.1f}s')
