import numpy as np, pandas as pd, json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
X = pd.read_csv('X.csv').values
maps=["mod(2*x10+2*x23+x25+x30,1)","mod(2*x1-2*x6-x11-x24,1)"]
def M(X):
    x=lambda k:X[:,k-1]
    return np.column_stack([(2*x(10)+2*x(23)+x(25)+x(30))%1,(2*x(1)-2*x(6)-x(11)-x(24))%1])
def fit(P):
    c=np.array([.517,.511]); cent=np.deg2rad([19,142,-98.])
    for it in range(10):
        D=P-c; ang=np.arctan2(D[:,1],D[:,0]); rad=np.hypot(*D.T)
        lab=np.argmin(np.abs(((ang[:,None]-cent[None]+np.pi)%(2*np.pi))-np.pi),1); L=[]
        for k in range(3):
            Q=P[(rad>.02)&(lab==k)]; m=Q.mean(0); dv=np.linalg.svd(Q-m)[2][0]
            if dv@(m-c)<0: dv=-dv
            L.append((m,dv)); cent[k]=np.arctan2(dv[1],dv[0])
        A=np.array([[-dv[1],dv[0]] for m,dv in L]); b=np.array([a@m for a,(m,dv) in zip(A,L)])
        c=np.linalg.lstsq(A,b,rcond=None)[0]
    ends=[]
    for k,(m,dv) in enumerate(L):
        t=((P-c)@dv)[(lab==k)&(rad>.02)]
        # arm end = where density drops to half its plateau value (removes blur from thickness)
        h,e=np.histogram(t,60,(0,.15)); plat=np.median(h[8:32]); i=np.where(h>plat/2)[0].max()
        ends.append(c+dv*e[i+1])
    return c,np.array(ends)
def draw(c,ends,m=20):
    pts=[list(c)]; seg=[]
    for e in ends:
        prev=0
        for s in range(1,m+1):
            pts.append(list(c+(e-c)*s/m)); seg.append([prev,len(pts)-1]); prev=len(pts)-1
    return np.array(pts),seg
def dist(P,pts,seg,scale):
    best=np.full(len(P),np.inf)
    for i,j in seg:
        a,b=pts[i]/scale,pts[j]/scale; ab=b-a; Q=P/scale
        t=np.clip(((Q-a)@ab)/(ab@ab),0,1); best=np.minimum(best,np.hypot(*(Q-a-t[:,None]*ab).T))
    return best
rng=np.random.default_rng(0); noise=M(rng.random((5000,30))); scale=noise.std(0)
# holdout check
perm=rng.permutation(5000); tr,te=perm[:4000],perm[4000:]
c,ends=fit(M(X[tr])); pts,seg=draw(c,ends)
dte=dist(M(X[te]),pts,seg,scale); dno=dist(noise,pts,seg,scale)
print('holdout: fitted on 4000, tested on 1000. scaled distance median/95%/max:',np.quantile(dte,[.5,.95,1]).round(4))
print('noise rows scaled distance 5%/median:',np.quantile(dno,[.05,.5]).round(3),' frac noise within 0.06:',(dno<0.06).mean(),' frac holdout within 0.06:',(dte<0.06).mean())
# final fit on all rows
P=M(X); c,ends=fit(P); pts,seg=draw(c,ends)
print('junction',c.round(4),'ends',ends.round(4),'lengths',np.hypot(*(ends-c).T).round(4))
report=("Structure found. 22 of the 30 columns are pure uniform noise; the structure lives in 8 columns, in two groups of four, and is only visible "
"through mod-1 integer combinations: u=(2*x10+2*x23+x25+x30) mod 1 and v=(2*x1-2*x6-x11-x24) mod 1. On noise both are uniform on [0,1); on the data, "
"(u,v) lies on a three-armed 'Y' (a 1-D branching skeleton): junction at about (%.3f,%.3f), three straight arms ending at about (%.3f,%.3f), (%.3f,%.3f) and (%.3f,%.3f) "
"(directions ~19, ~142, ~-98 degrees, lengths ~0.10-0.12), points spread evenly along the arms with perpendicular scatter sd ~0.005. The shape sits near the middle of the unit square, so it does not cross the wrap-around border. "
"How: marginals, correlations, PCA, pairwise/triple histograms and Fourier characters were all null. A scan of all 4-column subsets (chi-square histograms and torus nearest-neighbour distances) flagged exactly two subsets "
"(x10,x23,x25,x30) and (x1,x6,x11,x24) at z~40-57 (all others |z|<5). A Fourier-character search over integer coefficients inside each gave the combinations (2,2,1,1) and (2,-2,-1,-1) with resultant length ~0.96; "
"plotting one against the other revealed the Y. No third coordinate found: position along the Y is independent of other columns' low-order characters, and a nearest-neighbour scan of all other 5-column subsets was null (relations needing 6+ columns were not scanned). "
"A fit on 4000 rows reproduces on the 1000 held-out rows.") % (c[0],c[1],*ends.ravel())
sub={"status":"structure","map":maps,"drawing":{"points":[[round(float(a),5),round(float(b),5)] for a,b in pts],"segments":seg,"triangles":[]},"report":report}
assert len(pts)<=200 and all(len(m)<=200 for m in maps)
json.dump(sub,open('submission.json','w'),indent=1)
fig,ax=plt.subplots(1,2,figsize=(15,7))
ax[0].scatter(noise[:,0],noise[:,1],s=2,c='0.7',label='pure noise rows (simulated)'); ax[0].scatter(P[:,0],P[:,1],s=2,c='C0',label='X.csv')
ax[0].set_xlim(0,1); ax[0].set_ylim(0,1); ax[0].set_aspect('equal'); ax[0].legend(loc='upper right'); ax[0].set_title('map output: data vs noise'); ax[0].set_xlabel(maps[0]); ax[0].set_ylabel(maps[1])
ax[1].scatter(P[:,0],P[:,1],s=3,c='C0',alpha=.35); ax[1].plot(pts[:,0],pts[:,1],'r.',ms=4)
for i,j in seg: ax[1].plot(pts[[i,j],0],pts[[i,j],1],'r-',lw=1)
ax[1].set_aspect('equal'); ax[1].set_title('zoom: data with submitted drawing (red)'); ax[1].set_xlabel('u'); ax[1].set_ylabel('v')
plt.tight_layout(); plt.savefig('plots/09_final.png',dpi=100)
