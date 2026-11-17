import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
X = pd.read_csv('X.csv').values
x=lambda k:X[:,k-1]
u=(2*x(10)+2*x(23)+x(25)+x(30))%1
v=(2*x(1)-2*x(6)-x(11)-x(24))%1
P=np.column_stack([u,v])
# initial centre: density mode
c=np.array([0.51,0.515])
for it in range(20):
    D=P-c; ang=np.arctan2(D[:,1],D[:,0]); rad=np.hypot(*D.T)
    far=rad>0.02
    # cluster angles into 3 arms
    if it==0:
        h,e=np.histogram(ang[far],72,(-np.pi,np.pi)); print('angle hist',h)
        cent=np.array([np.deg2rad(a) for a in (15,150,-95)])
    lab=np.argmin(np.abs(((ang[:,None]-cent[None]+np.pi)%(2*np.pi))-np.pi),1)
    lines=[]
    for k in range(3):
        Q=P[far&(lab==k)]; m=Q.mean(0); U,S,Vt=np.linalg.svd(Q-m); dvec=Vt[0]
        if dvec@(m-c)<0: dvec=-dvec
        lines.append((m,dvec)); cent[k]=np.arctan2(dvec[1],dvec[0])
    # centre = least squares intersection of 3 lines
    A=[];b=[]
    for m,dv in lines:
        nrm=np.array([-dv[1],dv[0]]); A.append(nrm); b.append(nrm@m)
    c=np.linalg.lstsq(np.array(A),np.array(b),rcond=None)[0]
print('centre',c)
D=P-c; ang=np.arctan2(D[:,1],D[:,0]); rad=np.hypot(*D.T)
lab=np.argmin(np.abs(((ang[:,None]-cent[None]+np.pi)%(2*np.pi))-np.pi),1)
ends=[]
for k,(m,dv) in enumerate(lines):
    sel=(lab==k)&(rad>0.02); t=(P[sel]-c)@dv; nrm=np.array([-dv[1],dv[0]]); r=(P[sel]-c)@nrm
    print(f'arm{k}: angle {np.rad2deg(cent[k]):.1f} deg, n={sel.sum()}, t quantiles {np.quantile(t,[.5,.95,.99,.999,1]).round(4)}, perp sd {r.std():.4f}, perp mean {r.mean():.5f}')
    # curvature check: perp residual vs t
    print('   perp mean by t-quartile', [round(float(r[(t>=a)&(t<b)].mean()),4) for a,b in zip(np.quantile(t,[0,.25,.5,.75]),np.quantile(t,[.25,.5,.75,1]))],
          ' perp sd by quartile',[round(float(r[(t>=a)&(t<b)].std()),4) for a,b in zip(np.quantile(t,[0,.25,.5,.75]),np.quantile(t,[.25,.5,.75,1]))])
    print('   t hist', np.histogram(t,10,(0,0.12))[0])
    ends.append(c+dv*np.quantile(t,.995))
print('near centre (rad<0.02):',(rad<0.02).sum(), ' frac', (rad<0.02).mean())
print('rad hist all', np.histogram(rad,12,(0,0.12))[0])
print('ends',np.array(ends))
np.save('scripts/yfit.npy',np.vstack([c]+ends))
fig,ax=plt.subplots(figsize=(8,8))
ax.scatter(u,v,s=3,alpha=.4,c=lab,cmap='viridis')
for e in ends: ax.plot([c[0],e[0]],[c[1],e[1]],'r-',lw=1.5)
ax.set_aspect('equal'); ax.set_xlabel('u = (2x10+2x23+x25+x30) mod 1'); ax.set_ylabel('v = (2x1-2x6-x11-x24) mod 1'); ax.set_title('Y structure, zoomed, with fitted skeleton')
plt.savefig('plots/07_Y_zoom.png',dpi=110)
