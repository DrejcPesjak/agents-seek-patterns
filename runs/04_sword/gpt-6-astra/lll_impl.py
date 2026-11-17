import numpy as np

def lll(A,delta=.75,maxsteps=100000):
 A=np.array(A,dtype=object);n=len(A);mu=np.zeros((n,n),np.longdouble);bs=np.zeros_like(mu);B=np.zeros(n,np.longdouble)
 for i in range(n):
  v=np.array(A[i],np.longdouble)
  for _ in range(2):
   for j in range(i):
    q=np.dot(v,bs[j])/B[j];mu[i,j]+=q;v-=q*bs[j]
  bs[i]=v;B[i]=np.dot(v,v)
 k=1;steps=0
 while k<n:
  steps+=1
  if steps>maxsteps:raise RuntimeError('LLL step limit')
  for j in range(k-1,-1,-1):
   q=int(np.rint(mu[k,j]))
   if q:
    A[k]-=q*A[j];mu[k,:j]-=q*mu[j,:j];mu[k,j]-=q
  if B[k]>=(delta-mu[k,k-1]**2)*B[k-1]:k+=1
  else:
   old=mu[k,k-1];bb=B[k]+old**2*B[k-1];new=old*B[k-1]/bb
   B[k]=B[k-1]*B[k]/bb;B[k-1]=bb;mu[k,k-1]=new
   A[[k,k-1]]=A[[k-1,k]]
   for j in range(k-1):mu[k,j],mu[k-1,j]=mu[k-1,j],mu[k,j]
   for i in range(k+1,n):
    t=mu[i,k];mu[i,k]=mu[i,k-1]-old*t;mu[i,k-1]=t+new*mu[i,k]
   k=max(k-1,1)
 return np.array(A,dtype=np.int64)
