#!/usr/bin/env python3
"""Reproduce with python generator.py --reproduce. Requires numpy, scipy, sklearn.
Reference solver sees only X and the frequency search class, not B or shape."""
import os
for v in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'): os.environ[v]='4'
import argparse,itertools,json,time,secrets
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import shortest_path, connected_components
from sklearn.cluster import KMeans
from grade import grade, EPSILON, load, Invalid
ROOT=Path(__file__).resolve().parent
D=30; N=5000; NOISE=.005; NULLS=39
SHAPES=('X of crossing segments','Y-shaped tree','two separate loops','figure-8','random tree with 4–6 nodes')
def make_shape(rng):
    name=str(rng.choice(SHAPES)); V=[]; E=[]
    if name==SHAPES[0]:
        V=np.array([[0,0],[-.20,-.16],[.20,.16],[-.19,.18],[.19,-.18]]); E=[(0,j) for j in range(1,5)]
    elif name==SHAPES[1]:
        theta=rng.uniform(0,2*np.pi)+np.arange(3)*2*np.pi/3+rng.uniform(-.16,.16,3)
        V=np.vstack([np.zeros((1,2)),np.column_stack((np.cos(theta),np.sin(theta)))*rng.uniform(.17,.25,(3,1))]); E=[(0,j) for j in range(1,4)]
    elif name==SHAPES[2]:
        for side in (-1,1):
            t=np.arange(96)*2*np.pi/96; start=len(V)
            V.extend(np.column_stack((side*.125+.085*np.cos(t),.10*np.sin(t))))
            E.extend((start+i,start+(i+1)%96) for i in range(96))
    elif name==SHAPES[3]:
        V=[[0.,0.]]
        for side in (-1,1):
            t=np.arange(1,81)*2*np.pi/81; start=len(V)
            V.extend(np.column_stack((side*.115*(1-np.cos(t)),.125*np.sin(t))))
            E.extend([(0,start)]+[(j,j+1) for j in range(start,start+79)]+[(start+79,0)])
    else:
        n=int(rng.integers(4,7)); angles=np.arange(n-1)*2*np.pi/(n-1)+rng.uniform(-.2,.2,n-1)
        V=[[0.,0.]]; E=[]
        for j in range(1,n):
            parent=0 if j<4 else int(rng.integers(1,j))
            end=np.array(V[parent])+rng.uniform(.09,.17)*np.array([np.cos(angles[j-1]),np.sin(angles[j-1])])
            V.append(end.tolist()); E.append((parent,j))
    V=np.array(V,dtype=float); E=np.array(E,dtype=int)
    ang=rng.uniform(0,2*np.pi); rot=np.array([[np.cos(ang),-np.sin(ang)],[np.sin(ang),np.cos(ang)]])
    V=V@rot.T; V*=rng.uniform(.85,1.05,2); V*=min(1.,.29/np.max(np.abs(V)))
    V+=rng.uniform(.46,.54,2)
    return name,V,E

def sample_u(rng,V,E,n,noise=NOISE):
    lens=np.linalg.norm(V[E[:,0]]-V[E[:,1]],axis=1); ids=rng.choice(len(E),size=n,p=lens/lens.sum()); t=rng.random((n,1))
    return (V[E[ids,0]]*(1-t)+V[E[ids,1]]*t+rng.normal(0,noise,(n,2)))%1

def weights(rng):
    B=np.zeros((2,D),dtype=int); places=rng.permutation(D)[:8].reshape(2,4)
    for row,p in enumerate(places): B[row,p]=rng.permutation([1,1,2,2])*rng.choice([-1,1],4)
    return B

def embed(rng,B,u):
    x=rng.random((len(u),D)); p=[int(np.flatnonzero(np.abs(B[i])==1)[0]) for i in range(2)]
    x[:,p]=0; x[:,p]=((u-x@B.T)@np.linalg.inv(B[:,p]).T)%1
    assert np.max(np.abs(((x@B.T-u+.5)%1)-.5))<1e-12
    return x,p

def diagnostic(x):
    corr=np.corrcoef(x,rowvar=False); maxcorr=np.abs(corr[np.triu_indices(D,1)]).max()
    eig=np.linalg.eigvalsh(np.cov(x,rowvar=False)); p8=np.floor(8*x).astype(np.int32).T; p4=np.floor(4*x).astype(np.int32).T
    pair=0.; triple=0.
    for i,j in itertools.combinations(range(D),2):
        cnt=np.bincount(p8[i]*8+p8[j],minlength=64); pair=max(pair,float(((cnt-N/64)**2).sum()/(N/64)))
    for i,j,k in itertools.combinations(range(D),3):
        cnt=np.bincount(p4[i]*16+p4[j]*4+p4[k],minlength=64); triple=max(triple,float(((cnt-N/64)**2).sum()/(N/64)))
    return [float(maxcorr),float(eig[-1]/eig[0]),float(np.sqrt(np.mean((eig/(1/12)-1)**2))),pair,triple]

def conceal(x,seed):
    rng=np.random.default_rng(seed); a=np.array([diagnostic(x)]+[diagnostic(rng.random((N,D))) for _ in range(NULLS)])
    ranks=np.sum(a[None,:,:]>=a[:,None,:],axis=1)/(NULLS+1); minp=ranks.min(axis=1); omni=float(np.mean(minp<=minp[0])); p=ranks[0]
    return {'names':['max_abs_corr','PCA_max_min_eigenvalue_ratio','PCA_spectrum_RMS_relative','max_pair_chi2_8x8','max_triple_chi2_4x4x4'], 'observed':a[0].tolist(),'null_median':np.median(a[1:],axis=0).tolist(),'p_values':p.tolist(),'omnibus_p':omni,'null_count':NULLS,'pass':bool(omni>.05 and np.all(p>.05))}

def candidates(cols):
    index=np.array(list(itertools.combinations(cols,2)),int); co=np.array(list(itertools.product((-2,-1,1,2),repeat=2)),int)
    return np.repeat(index,16,axis=0),np.tile(co,(len(index),1))

def features(x,ii,cc): return np.exp(2j*np.pi*(x[:,ii[:,0]]*cc[:,0]+x[:,ii[:,1]]*cc[:,1])).T.astype(np.complex64)

def solve_lattice(x,seed=917):
    rng=np.random.default_rng(seed); found={}; checks=0; t0=time.process_time(); wall=time.perf_counter()
    for attempt in range(24):
        p=rng.permutation(D); ii,cc=candidates(p[:15]); jj,dd=candidates(p[15:]); screen=x[rng.choice(len(x),512,replace=False)]
        L=features(screen,ii,cc); R=features(screen,jj,dd); C=np.abs(L@R.T)/len(screen)
        hits=np.argwhere(C>.20); checks+=len(hits)
        for i,j in hits:
            a=np.zeros(D,dtype=int); a[ii[i]]=cc[i]; a[jj[j]]=dd[j]; a*=np.sign(a[np.flatnonzero(a)[0]])
            if np.gcd.reduce(np.abs(a))!=1: continue
            key=tuple(a)
            if key in found: continue
            rho=float(abs(np.mean(np.exp(2j*np.pi*(x@a)))))
            if rho>=.28: found[key]=rho
        if len(found)>=2 and np.linalg.matrix_rank(np.array(list(found)))>=2: break
    if len(found)<2: raise RuntimeError('reference solver failed')
    a=sorted(found,key=found.get,reverse=True); rows=[]
    for key in a:
        if not rows or np.linalg.matrix_rank(np.array(rows+[key]))>len(rows): rows.append(key)
        if len(rows)==2: break
    return np.array(rows),{'partitions':attempt+1,'screen_candidates_verified':checks,'cpu_seconds':time.process_time()-t0,'wall_seconds':time.perf_counter()-wall,'magnitudes':[found[tuple(r)] for r in rows]}
def fit_drawing(u):
    # Compact point/segment skeleton inferred from visible points, not generator.
    t0=time.process_time(); K=100; km=KMeans(n_clusters=K,random_state=9821,n_init=3).fit(u); V=km.cluster_centers_
    dist=cdist(V,V); links=cdist(u,V); nearest=np.argpartition(links,1,axis=1)[:,:2]; count=np.zeros((K,K),int)
    for i,j in nearest: count[min(i,j),max(i,j)]+=1
    candidates=[(float(dist[i,j]),i,j) for i,j in itertools.combinations(range(K),2) if dist[i,j]<.045 and count[i,j]>=2]
    par=list(range(K))
    def root(i):
        while par[i]!=i: par[i]=par[par[i]]; i=par[i]
        return i
    E=[]; extra=[]
    for d,i,j in sorted(candidates):
        a,b=root(i),root(j)
        if a!=b: par[a]=b; E.append((i,j))
        else: extra.append((d,i,j))
    G=np.full((K,K),np.inf); np.fill_diagonal(G,0)
    for i,j in E: G[i,j]=G[j,i]=dist[i,j]
    for d,i,j in extra:
        path=shortest_path(G,directed=False,indices=i)[j]
        if path>max(5*d,.12): E.append((i,j)); G[i,j]=G[j,i]=d
    return {'points':V.tolist(),'segments':[list(e) for e in E],'triangles':[]}, {'cpu_seconds':time.process_time()-t0,'vertices':K,'edges':len(E)}
def expr(a):
    terms=[]
    for j,v in enumerate(a):
        if v: terms.append(('+' if v>0 else '-')+(str(abs(v))+'*' if abs(v)>1 else '')+f'x{j+1}')
    return '('+''.join(terms).lstrip('+')+')%1'
def cert(A,dr,report): return {'status':'structure','map':[expr(a) for a in A],'drawing':dr,'report':report}
def dump(path,obj): path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def test_submissions(ref,A):
    dr=ref['drawing']; P=np.array(dr['points']); eqA=A.copy(); eqA[0]*=-1; eqP=P.copy(); eqP[:,0]=(-eqP[:,0])%1
    assert np.max(np.abs(eqP[:,0]-(1-P[:,0])))<1e-12 and P[:,0].min()>.03 and P[:,0].max()<.97
    eq=cert(eqA,{**dr,'points':eqP.tolist()},'Equivalent reflected coordinate and drawing.')
    square={'points':[[0,0],[1,0],[1,1],[0,1]],'segments':[],'triangles':[[0,1,2],[0,2,3]]}
    def c(m,d): return {'status':'structure','map':m,'drawing':d,'report':'Diagnostic submission.'}
    lo,hi=P[:,0].min(),P[:,0].max()
    return {'reference':ref,'equivalent':eq,'one_direction':c(ref['map'][:1],{'points':[[float(lo)],[float(hi)]],'segments':[[0,1]],'triangles':[]}), 'constant':c(['0'],{'points':[[0]],'segments':[],'triangles':[]}), 'x1_x2':c(['x1','x2'],dr), 'filled_square':c(ref['map'],square), 'over_200':c(ref['map'],{'points':[[.5,.5]]*201,'segments':[],'triangles':[]}), 'none':{'status':'none','map':[],'drawing':{'points':[],'segments':[],'triangles':[]},'report':'No structure.'}}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int); ap.add_argument('--reproduce',action='store_true'); args=ap.parse_args()
    seed=json.loads((ROOT/'answer_key.json').read_text())['seed'] if args.reproduce else (args.seed if args.seed is not None else secrets.randbits(63))
    for attempt in range(30):
        streams=np.random.SeedSequence([seed,attempt]).spawn(7); rngs=[np.random.default_rng(s) for s in streams]
        name,V,E=make_shape(rngs[0]); B=weights(rngs[1]); U=sample_u(rngs[2],V,E,N); X,p=embed(rngs[3],B,U)
        rhos=np.abs(np.exp(2j*np.pi*(X@B.T)).mean(axis=0))
        if np.min(rhos)<.3: continue
        diagnostics=conceal(X,int(rngs[4].integers(2**32))); print('concealment trial',attempt,diagnostics['pass'],flush=True)
        if not diagnostics['pass']: continue
        A,recovery=solve_lattice(X)
        if np.linalg.matrix_rank(np.vstack([A,B]))!=2: continue
        u=(X@A.T)%1; dr,fit=fit_drawing(u)
        G=np.zeros((len(dr['points']),len(dr['points'])),int)
        for i,j in dr['segments']: G[i,j]=G[j,i]=1
        comp=connected_components(G,directed=False,return_labels=False); loops=len(dr['segments'])-len(G)+comp
        report=f'Two sparse integer modular directions expose a thin geometric graph in the unit square. Fitted graph: {comp} components, {loops} independent cycles.'
        ref=cert(A,dr,report); private,_=embed(rngs[5],B,sample_u(rngs[5],V,E,N)); noise=rngs[6].random((N,D))
        test=grade(ref,private,noise)
        if test['hit_planted']<.9 or test['drawing_precision']<.9: continue
        header=','.join(f'x{i+1}' for i in range(D))
        for path,data in [(ROOT/'../agent/X.csv',X),(ROOT/'private_planted.csv',private),(ROOT/'private_noise.csv',noise)]: np.savetxt(path,data,delimiter=',',header=header,comments='',fmt='%.17g')
        private=load('private_planted.csv'); noise=load('private_noise.csv'); results={}
        for label,obj in test_submissions(ref,A).items():
            try: results[label]=grade(obj,private,noise)
            except Invalid as e: results[label]={'status':'rejected','reason':str(e),'score':0.}
        key={'seed':seed,'accepted_attempt_zero_based':attempt,'shape':name,'true_vertices':V.tolist(),'true_edges':E.tolist(),'B':B.tolist(),'pivot_columns_one_based':[i+1 for i in p],'delta':4,'saturated':True,'pivot_determinant':int(round(np.linalg.det(B[:,p]))),'latent_noise_sigma':NOISE,'visible_fourier_magnitudes':rhos.tolist(),'concealment':diagnostics,'recovered_B':A.tolist(),'recovery':{**recovery,'fit':fit,'stacked_rank':2,'full_lattice_recovered':True},'epsilon':EPSILON,'tests':results,'grader_note':'Shape score is geometry/detection, not a proof of lattice or topology recovery; report not scored.'}
        dump(ROOT/'answer_key.json',key); dump(ROOT/'reference_submission.json',ref)
        print(json.dumps({'attempts':attempt+1,'delta':4,'saturated':True,'abs_pivot_det':abs(key['pivot_determinant']),'rho_range':[float(min(rhos)),float(max(rhos))],'concealment':diagnostics,'recovery':recovery,'fit_seconds':fit['cpu_seconds'],'tests':results},indent=2),flush=True); return
    raise RuntimeError('no accepted instance in 30 attempts')
if __name__=='__main__': main()
