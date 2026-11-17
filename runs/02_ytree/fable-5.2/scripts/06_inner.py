import numpy as np, pandas as pd, itertools
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
X = pd.read_csv('X.csv').values
x=lambda k:X[:,k-1]
fig,ax=plt.subplots(2,4,figsize=(20,10))
G=[('A',(10,23,25,30),(1,1,1,1)),('B',(1,6,11,24),(1,-1,-1,-1))]
for r,(nm,c,s) in enumerate(G):
    a=(s[0]*x(c[0])+s[1]*x(c[1]))%1; b=(s[2]*x(c[2])+s[3]*x(c[3]))%1
    u=(2*a+b)%1
    ax[r,0].scatter(a,b,s=2,alpha=.5); ax[r,0].set_xlabel(f'a=({s[0]}*x{c[0]}+{s[1]}*x{c[1]})%1'); ax[r,0].set_ylabel(f'b=({s[2]}*x{c[2]}+{s[3]}*x{c[3]})%1')
    ax[r,1].scatter(a,u,s=2,alpha=.5); ax[r,1].set_xlabel('a'); ax[r,1].set_ylabel('u=(2a+b)%1')
    ax[r,2].scatter(b,u,s=2,alpha=.5); ax[r,2].set_xlabel('b'); ax[r,2].set_ylabel('u')
    ax[r,3].scatter((2*a)%1,u,s=2,alpha=.5); ax[r,3].set_xlabel('2a%1'); ax[r,3].set_ylabel('u')
    for k in range(4):
        ax[r,k].set_xlim(0,1)
plt.tight_layout(); plt.savefig('plots/06_inner.png',dpi=80)
# u vs each individual column
fig,ax=plt.subplots(2,4,figsize=(20,9))
for r,(nm,c,s) in enumerate(G):
    u=(2*s[0]*x(c[0])+2*s[1]*x(c[1])+s[2]*x(c[2])+s[3]*x(c[3]))%1
    for k in range(4):
        ax[r,k].scatter(x(c[k]),u,s=2,alpha=.5); ax[r,k].set_xlabel(f'x{c[k]}'); ax[r,k].set_ylabel('u' if r==0 else 'v')
plt.tight_layout(); plt.savefig('plots/06_u_vs_cols.png',dpi=80)
