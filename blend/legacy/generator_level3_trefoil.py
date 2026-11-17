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
D=30; N=8000; K=3; S=5; NOISE=.005; NULLS=39
def make_shape(rng):
    p,q=[(2,3),(2,5),(3,4)][int(rng.integers(3))]
    t=np.arange(192)*2*np.pi/192; R=rng.uniform(.15,.18); r=rng.uniform(.075,.09); phase=rng.uniform(0,2*np.pi)
    a=p*t; b=q*t+phase
    V=np.column_stack(((R+r*np.cos(b))*np.cos(a),(R+r*np.cos(b))*np.sin(a),r*np.sin(b)))
    rot,_=np.linalg.qr(rng.normal(size=(3,3))); V=V@rot.T; V*=rng.uniform(.94,1.06,3); V*=min(1.,.275/np.max(np.abs(V))); V+=rng.uniform(.47,.53,3)
    E=np.column_stack((np.arange(len(V)),np.roll(np.arange(len(V)),-1)))
    return f'torus knot ({p},{q})',V,E,{'winding_p':p,'winding_q':q,'major_radius':R,'minor_radius':r,'phase':phase}
def sample_u(rng,V,E,n,noise=NOISE):
    lens=np.linalg.norm(V[E[:,0]]-V[E[:,1]],axis=1); ids=rng.choice(len(E),size=n,p=lens/lens.sum()); t=rng.random((n,1))
    return (V[E[ids,0]]*(1-t)+V[E[ids,1]]*t+rng.normal(0,noise,(n,K)))%1

def weights(rng):
    B=np.zeros((K,D),dtype=int); places=rng.permutation(D)[:K*S].reshape(K,S)
    for row,p in enumerate(places): B[row,p]=rng.permutation([1,1,1,2,2])*rng.choice([-1,1],S)
    return B

def embed(rng,B,u):
    x=rng.random((len(u),D)); p=[int(np.flatnonzero(np.abs(B[i])==1)[0]) for i in range(K)]
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

def candidates(cols,s,canonical=False):
    index=np.array(list(itertools.combinations(sorted(cols),s)),int); co=np.array(list(itertools.product((-2,-1,1,2),repeat=s)),int)
    if canonical: co=co[co[:,0]>0]
    return np.repeat(index,len(co),axis=0),np.tile(co,(len(index),1))

def features(x,ii,cc):
    base=np.exp(2j*np.pi*x).astype(np.complex64); out=np.ones((len(ii),len(x)),dtype=np.complex64)
    for j in range(ii.shape[1]):
        w=base[:,ii[:,j]].T.copy(); ix=cc[:,j]<0; w[ix]=w[ix].conj(); ix=np.abs(cc[:,j])==2; w[ix]*=w[ix]; out*=w
    return out

def solve_lattice(x,seed=119):
    rng=np.random.default_rng(seed); found={}; checks=0; pairs=0; t0=time.process_time(); wall=time.perf_counter()
    for attempt in range(32):
        p=rng.permutation(D); screen=x[rng.choice(len(x),256,replace=False)]
        for left,right in ((p[:15],p[15:]),(p[15:],p[:15])):
            ii,cc=candidates(left,2,True); jj,dd=candidates(right,3)
            L=features(screen,ii,cc); R=features(screen,jj,dd)
            for start in range(0,len(R),1024):
                C=np.abs(L@R[start:start+1024].T)/len(screen); pairs+=C.size; hits=np.argwhere(C>.28); checks+=len(hits)
                for i,j0 in hits:
                    j=j0+start; a=np.zeros(D,dtype=int); a[ii[i]]=cc[i]; a[jj[j]]=dd[j]; a*=np.sign(a[np.flatnonzero(a)[0]])
                    if np.gcd.reduce(np.abs(a))!=1: continue
                    key=tuple(a)
                    if key in found: continue
                    rho=float(abs(np.mean(np.exp(2j*np.pi*(x@a)))))
                    if rho>=.30: found[key]=rho
            if len(found)>=K and np.linalg.matrix_rank(np.array(list(found)))>=K: break
        if len(found)>=K and np.linalg.matrix_rank(np.array(list(found)))>=K: break
    rows=[]
    for key in sorted(found,key=found.get,reverse=True):
        if not rows or np.linalg.matrix_rank(np.array(rows+[key]))>len(rows): rows.append(key)
        if len(rows)==K: break
    if len(rows)!=K: raise RuntimeError('reference solver failed')
    return np.array(rows),{'partitions':attempt+1,'screen_candidates_verified':checks,'candidate_pairs':pairs,'cpu_seconds':time.process_time()-t0,'wall_seconds':time.perf_counter()-wall,'magnitudes':[found[tuple(r)] for r in rows]}
def fit_drawing(u):
    # Compact point/segment skeleton inferred from visible points, not generator.
    t0=time.process_time(); K=160; km=KMeans(n_clusters=K,random_state=9821,n_init=3).fit(u); V=km.cluster_centers_
    dist=cdist(V,V); links=cdist(u,V); nearest=np.argpartition(links,1,axis=1)[:,:2]; count=np.zeros((K,K),int)
    for i,j in nearest: count[min(i,j),max(i,j)]+=1
    candidates=[(float(dist[i,j]),i,j) for i,j in itertools.combinations(range(K),2) if dist[i,j]<.065 and count[i,j]>=3]
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
        if path>max(6*d,.15): E.append((i,j)); G[i,j]=G[j,i]=d
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
    eq=cert(eqA,{**dr,'points':eqP.tolist()},'Equivalent reflected coordinate and drawing.')
    square={'points':[[0,0],[1,0],[1,1],[0,1]],'segments':[],'triangles':[[0,1,2],[0,2,3]]}
    def c(m,d): return {'status':'structure','map':m,'drawing':d,'report':'Diagnostic submission.'}
    lo,hi=P[:,0].min(),P[:,0].max()
    return {'reference':ref,'equivalent':eq,'one_direction':c(ref['map'][:1],{'points':[[float(lo)],[float(hi)]],'segments':[[0,1]],'triangles':[]}), 'constant':c(['0'],{'points':[[0]],'segments':[],'triangles':[]}), 'unrelated_coordinates':c(['x1','x2','x3'],dr), 'filled_square':c(ref['map'][:2],square), 'over_200':c(ref['map'],{'points':[[.5,.5,.5]]*201,'segments':[],'triangles':[]}), 'none':{'status':'none','map':[],'drawing':{'points':[],'segments':[],'triangles':[]},'report':'No structure.'}}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int); ap.add_argument('--reproduce',action='store_true'); args=ap.parse_args()
    seed=json.loads((ROOT/'answer_key.json').read_text())['seed'] if args.reproduce else (args.seed if args.seed is not None else secrets.randbits(63))
    for attempt in range(30):
        streams=np.random.SeedSequence([seed,attempt]).spawn(7); rngs=[np.random.default_rng(s) for s in streams]
        name,V,E,params=make_shape(rngs[0]); B=weights(rngs[1]); U=sample_u(rngs[2],V,E,N); X,p=embed(rngs[3],B,U)
        rhos=np.abs(np.exp(2j*np.pi*(X@B.T)).mean(axis=0))
        if np.min(rhos)<.35: continue
        diagnostics=conceal(X,int(rngs[4].integers(2**32))); print('concealment trial',attempt,diagnostics['pass'],flush=True)
        if not diagnostics['pass']: continue
        A,recovery=solve_lattice(X); print("reference directions recovered",flush=True)
        if np.linalg.matrix_rank(np.vstack([A,B]))!=K: continue
        u=(X@A.T)%1; dr,fit=fit_drawing(u)
        G=np.zeros((len(dr['points']),len(dr['points'])),int)
        for i,j in dr['segments']: G[i,j]=G[j,i]=1
        comp=connected_components(G,directed=False,return_labels=False); loops=len(dr['segments'])-len(G)+comp
        report=f'Three sparse integer modular directions expose a thin geometric curve in the unit cube. Fitted graph: {comp} components, {loops} independent cycles.'
        ref=cert(A,dr,report); private,_=embed(rngs[5],B,sample_u(rngs[5],V,E,5000)); noise=rngs[6].random((5000,D))
        test=grade(ref,private,noise)
        print('geometry gate',test['hit_planted'],test['drawing_precision'],comp,loops,flush=True)
        if test['hit_planted']<.9 or test['drawing_precision']<.9 or comp!=1 or loops!=1 or not np.all(G.sum(axis=0)==2): continue
        header=','.join(f'x{i+1}' for i in range(D))
        for path,data in [(ROOT/'../agent/X.csv',X),(ROOT/'private_planted.csv',private),(ROOT/'private_noise.csv',noise)]: np.savetxt(path,data,delimiter=',',header=header,comments='',fmt='%.17g')
        private=load('private_planted.csv'); noise=load('private_noise.csv'); results={}
        for label,obj in test_submissions(ref,A).items():
            try: results[label]=grade(obj,private,noise)
            except Invalid as e: results[label]={'status':'rejected','reason':str(e),'score':0.}
        Ub=np.array([[int(np.array_equal(a,b))-int(np.array_equal(a,-b)) for b in B] for a in A]); assert np.array_equal(Ub@B,A) and abs(round(np.linalg.det(Ub)))==1
        from grade import distances
        true=(V@Ub.T)%1; pts=np.array(dr['points']); ed=np.array(dr['segments']); t=np.arange(20)/20
        dense=(true[E[:,0],None,:]*(1-t[None,:,None])+true[E[:,1],None,:]*t[None,:,None]).reshape(-1,3)
        tr=distances(dense,pts,ed.tolist(),[]); fitpts=(pts[ed[:,0],None,:]*(1-t[None,:,None])+pts[ed[:,1],None,:]*t[None,:,None]).reshape(-1,3); pr=distances(fitpts,true,E.tolist(),[])
        truth_metrics={'tolerance_raw':.02,'precision':float((pr<=.02).mean()),'recall':float((tr<=.02).mean()),'max_true_to_recovered':float(tr.max()),'max_recovered_to_true':float(pr.max())}
        if min(truth_metrics['precision'],truth_metrics['recall'])<.95: continue
        key={'seed':seed,'accepted_attempt_zero_based':attempt,'shape':name,'shape_parameters':params,'true_vertices':V.tolist(),'true_edges':E.tolist(),'B':B.tolist(),'pivot_columns_one_based':[i+1 for i in p],'delta':S,'saturated':True,'pivot_determinant':int(round(np.linalg.det(B[:,p]))),'latent_noise_sigma':NOISE,'visible_fourier_magnitudes':rhos.tolist(),'concealment':diagnostics,'recovered_B':A.tolist(),'reference_basis_from_true':Ub.tolist(),'recovery':{**recovery,'fit':fit,'stacked_rank':K,'full_lattice_recovered':True,'geometry_truth_check':truth_metrics,'components':int(comp),'cycles':int(loops)},'epsilon':EPSILON,'tests':results,'grader_note':'Unchanged Level 2 grader; does not directly score lattice or topology.'}
        dump(ROOT/'answer_key.json',key); dump(ROOT/'reference_submission.json',ref)
        print(json.dumps({'attempts':attempt+1,'delta':S,'saturated':True,'abs_pivot_det':abs(key['pivot_determinant']),'rho_range':[float(min(rhos)),float(max(rhos))],'concealment':diagnostics,'recovery':recovery,'fit_seconds':fit['cpu_seconds'],'tests':results},indent=2),flush=True); return
    raise RuntimeError('no accepted instance in 30 attempts')
if __name__=='__main__': main()
