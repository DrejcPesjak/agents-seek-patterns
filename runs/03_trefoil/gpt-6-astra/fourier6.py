import numpy as np,itertools,time
A=np.load('split.npz');tr=A['train'][:2000];te=A['test'];n,d=tr.shape
logfile=open('analysis.log','a')
def log(s):print(s,flush=True);logfile.write(str(s)+'\n');logfile.flush()
t=time.time();labs=[]
for support in [1,2,3]:
 for cols in itertools.combinations(range(d),support):
  for ks in itertools.product([-1,1],repeat=support-1):
   labs.append(tuple(zip(cols,(1,)+ks)))
F=np.empty((n,len(labs)),dtype=np.complex64)
for j,lab in enumerate(labs):F[:,j]=np.exp(2j*np.pi*sum(tr[:,col]*k for col,k in lab))
log(f'Fourier search supports up to 6, +/-1 coefficients: {len(labs)} features on {n} train rows.')
hits={};p=F.shape[1]
for start in range(0,p,256):
 C=F[:,start:start+256].T@F.conj()/n
 for ii,row in enumerate(C):
  i=start+ii
  for j in np.flatnonzero(abs(row)>.105):
   if j>=i:continue
   dic={}
   for col,k in labs[i]:dic[col]=dic.get(col,0)+k
   for col,k in labs[j]:dic[col]=dic.get(col,0)-k
   lab=tuple((col,k) for col,k in sorted(dic.items()) if k)
   if not lab:continue
   if lab[0][1]<0:lab=tuple((col,-k) for col,k in lab)
   hits[lab]=max(hits.get(lab,0),float(abs(row[j])))
 if start%2048==0:log(f'Fourier6 progress {start}/{p}, elapsed {time.time()-t:.1f}s, candidates {len(hits)}')
ranked=sorted(hits.items(),key=lambda z:-z[1]);log(f'Fourier6 candidates {len(ranked)}')
for lab,score in ranked[:100]:
 val=abs(np.exp(2j*np.pi*sum(te[:,col]*k for col,k in lab)).mean())
 log(f'Fourier6 {tuple((col+1,k) for col,k in lab)}: train {score:.5f}, test {val:.5f}')
np.save('fourier6_hits.npy',np.array(ranked,dtype=object),allow_pickle=True)
log(f'Fourier6 elapsed {time.time()-t:.1f}s')
