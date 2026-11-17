#!/usr/bin/env python3
"""SEEK grader: score a submission.json against held-out planted and null rows.

Run:  python -m seek.grade submission.json --private path/to/private

The private directory must hold private_planted.csv and private_noise.csv
(plain or .gz) and may hold answer_key.json with a "grader_epsilon".

score = max(0, hit_planted - hit_noise) * drawing_precision

Epsilon is dimensionless: every map output is divided by its standard deviation
on the null rows. Drawing precision pools all vertices, 32 midpoint-grid samples
per unique segment and 64 deterministic uniform-area samples per unique triangle.
All distances are Euclidean, including at the square boundary.

Optional "unmix" key (meant for full-BLEND data): a matrix with one row per new
variable; z_i = unmix[i] . (x1..xM) and map expressions may use z1, z2, ...
Dependencies: numpy, scipy.
"""
import argparse, ast, json, math, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

MAX_UNMIX_ROWS = 256
# Preselected tolerance; reference must pass the >=90% hit-rate gate.
EPSILON = 0.055
FUN = {'mod':(np.mod,2),'abs':(np.abs,1),'floor':(np.floor,1),
       'sqrt':(np.sqrt,1),'sin':(np.sin,1),'cos':(np.cos,1),'atan2':(np.arctan2,2)}
OPS = {ast.Add:np.add, ast.Sub:np.subtract, ast.Mult:np.multiply,
       ast.Div:np.divide, ast.Mod:np.mod}
VARS = {}  # Populated per call from the dataset's column count (and unmix rows)
class Invalid(ValueError): pass

def parse(s):
    if not isinstance(s,str) or not 1 <= len(s) <= 200: raise Invalid('map expression must have 1–200 characters')
    try: t=ast.parse(s.strip(),mode='eval')
    except (SyntaxError,ValueError) as e: raise Invalid('invalid expression') from e
    def check(n):
        if isinstance(n,ast.Constant) and type(n.value) in (int,float):
            try: v=float(n.value)
            except (ValueError,OverflowError): raise Invalid('invalid constant')
            if not math.isfinite(v): raise Invalid('nonfinite constant')
        elif isinstance(n,ast.Name) and (n.id in VARS or n.id=='pi'): pass
        elif isinstance(n,ast.UnaryOp) and type(n.op) in (ast.UAdd,ast.USub): check(n.operand)
        elif isinstance(n,ast.BinOp) and type(n.op) in OPS: check(n.left); check(n.right)
        elif isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in FUN and not n.keywords and len(n.args)==FUN[n.func.id][1]:
            for a in n.args: check(a)
        else: raise Invalid('expression contains a forbidden operation/name')
    check(t.body)
    return t.body

def apply(n,X):
    if isinstance(n,ast.Constant): return np.float64(n.value)
    if isinstance(n,ast.Name): return np.pi if n.id=='pi' else X[:,VARS[n.id]]
    if isinstance(n,ast.UnaryOp):
        a=apply(n.operand,X); return -a if isinstance(n.op,ast.USub) else a
    if isinstance(n,ast.BinOp): return OPS[type(n.op)](apply(n.left,X),apply(n.right,X))
    if isinstance(n,ast.Call): return FUN[n.func.id][0](*(apply(a,X) for a in n.args))
    raise Invalid('invalid AST')

def mapped(trees,X):
    with np.errstate(all='ignore'):
        v=np.column_stack([np.broadcast_to(apply(t,X),(len(X),)) for t in trees]).astype(float)
    if not np.isfinite(v).all(): raise Invalid('nonfinite map output')
    return v

def validate(obj):
    if not isinstance(obj,dict) or obj.get('status') not in ('structure','none'): raise Invalid('bad status')
    if 'report' in obj and not isinstance(obj['report'],str): raise Invalid('report must be a string')
    if obj['status']=='none': return None
    if not isinstance(obj.get('report'),str): raise Invalid('report required')
    m=obj.get('map'); dr=obj.get('drawing')
    if not isinstance(m,list) or not 1<=len(m)<=3: raise Invalid('map must contain 1–3 expressions')
    trees=[parse(s) for s in m]; d=len(m)
    if not isinstance(dr,dict) or set(dr)!={'points','segments','triangles'}: raise Invalid('drawing requires points, segments, triangles')
    pts=dr['points']
    if not isinstance(pts,list) or not 1<=len(pts)<=200: raise Invalid('drawing must contain 1–200 points')
    for p in pts:
        if not isinstance(p,list) or len(p)!=d or any(type(v) not in (int,float) for v in p): raise Invalid('invalid point dimension/type')
    try: P=np.asarray(pts,dtype=float)
    except (OverflowError,ValueError): raise Invalid('invalid coordinates')
    if not np.isfinite(P).all(): raise Invalid('nonfinite drawing coordinates')
    def faces(name,k):
        a=dr[name]
        if not isinstance(a,list): raise Invalid(f'{name} must be a list')
        out=set()
        for f in a:
            if not isinstance(f,list) or len(f)!=k or any(type(i) is not int or not 0<=i<len(P) for i in f) or len(set(f))!=k: raise Invalid(f'invalid {name} indices')
            out.add(tuple(sorted(f)))
        return sorted(out)
    return trees,P,faces('segments',2),faces('triangles',3)

def seg_d2(X,a,b):
    v=b-a; den=v@v
    if den<=1e-28: return np.sum((X-a)**2,axis=1)
    t=np.clip(((X-a)@v)/den,0,1)
    return np.sum((X-a-t[:,None]*v)**2,axis=1)

def tri_d2(X,a,b,c):
    u=b-a; v=c-a; uu=u@u; vv=v@v; uv=u@v; det=uu*vv-uv*uv
    best=np.minimum.reduce([seg_d2(X,a,b),seg_d2(X,a,c),seg_d2(X,b,c)])
    if det>1e-14*max(uu*vv,1e-28):
        w=X-a; wu=w@u; wv=w@v
        s=(vv*wu-uv*wv)/det; t=(uu*wv-uv*wu)/det
        inside=(s>=0)&(t>=0)&(s+t<=1)
        dist=np.sum((w-s[:,None]*u-t[:,None]*v)**2,axis=1)
        best[inside]=np.minimum(best[inside],dist[inside])
    return best

def distances(X,P,E,T):
    out=cKDTree(P).query(X)[0]**2
    for i,j in E: out=np.minimum(out,seg_d2(X,P[i],P[j]))
    for i,j,k in T: out=np.minimum(out,tri_d2(X,P[i],P[j],P[k]))
    return np.sqrt(out)

def drawing_precision(P,E,T,A,epsilon):
    tree=cKDTree(A); hit=total=0
    def accumulate(q):
        nonlocal hit,total
        hit+=int(np.count_nonzero(tree.query(q)[0]<=epsilon)); total+=len(q)
    accumulate(P)
    t=(np.arange(32)+.5)/32
    rng=np.random.default_rng(4481)
    for i,j in E: accumulate(P[i]+t[:,None]*(P[j]-P[i]))
    for i,j,k in T:
        r=rng.random((64,2)); r[r.sum(axis=1)>1]=1-r[r.sum(axis=1)>1]
        accumulate(P[i]+r[:,0,None]*(P[j]-P[i])+r[:,1,None]*(P[k]-P[i]))
    return hit/total

def unmix_matrix(obj,M):
    W=obj.get('unmix') if isinstance(obj,dict) else None
    if W is None: return None
    if not isinstance(W,list) or not 1<=len(W)<=MAX_UNMIX_ROWS: raise Invalid(f'unmix must have 1–{MAX_UNMIX_ROWS} rows')
    for r in W:
        if not isinstance(r,list) or len(r)!=M or any(type(v) not in (int,float) for v in r): raise Invalid(f'every unmix row needs {M} numbers')
    try: W=np.asarray(W,dtype=float)
    except (OverflowError,ValueError): raise Invalid('invalid unmix values')
    if not np.isfinite(W).all() or np.max(np.abs(W))>1e6: raise Invalid('unmix values must be finite and at most 1e6 in size')
    return W

def grade(obj,planted,noise,epsilon=EPSILON):
    global VARS
    if planted.ndim!=2 or noise.ndim!=2 or planted.shape[1]!=noise.shape[1]:
        raise Invalid('planted/noise dimensions do not match')
    M=planted.shape[1]
    VARS = {f'x{i+1}':i for i in range(M)}
    W=unmix_matrix(obj,M)
    if W is not None:
        VARS.update({f'z{i+1}':M+i for i in range(len(W))})
        planted=np.hstack([planted,planted@W.T]); noise=np.hstack([noise,noise@W.T])
    parsed=validate(obj)
    zero={'hit_planted':0.,'hit_noise':0.,'drawing_precision':0.,'score':0.}
    if parsed is None: return {**zero,'status':'none','reason':'this benchmark instance contains structure'}
    trees,P,E,T=parsed
    A=mapped(trees,planted); Q=mapped(trees,noise)
    with np.errstate(all='ignore'):
        sd=Q.std(axis=0)
        if not np.isfinite(sd).all() or np.any(sd<1e-10): return {**zero,'status':'zero','reason':'near-zero or nonfinite null standard deviation'}
        A=A/sd; Q=Q/sd; P=P/sd
    if not all(np.isfinite(v).all() and np.max(np.abs(v))<1e140 for v in (A,Q,P)): raise Invalid('unsafe numeric range after normalization')
    hp=float(np.mean(distances(A,P,E,T)<=epsilon)); hn=float(np.mean(distances(Q,P,E,T)<=epsilon))
    pr=drawing_precision(P,E,T,A,epsilon)
    return {'status':'graded','epsilon':epsilon,'hit_planted':hp,'hit_noise':hn,'drawing_precision':pr,'score':max(0.,hp-hn)*pr}

def load(private,name):
    private=Path(private)
    for path in (private/name, private/(name+'.gz')):
        if path.exists(): return np.loadtxt(path,delimiter=',',skiprows=1)
    raise OSError(f'{name}[.gz] not found in {private}')

def grade_file(submission,private):
    """Grade a submission.json file against a private directory."""
    with open(submission,encoding='utf-8') as f: obj=json.load(f)
    key=Path(private)/'answer_key.json'
    epsilon=float(json.loads(key.read_text()).get('grader_epsilon',EPSILON)) if key.exists() else EPSILON
    return grade(obj,load(private,'private_planted.csv'),load(private,'private_noise.csv'),epsilon=epsilon)

def main():
    ap=argparse.ArgumentParser(description='Score a SEEK submission against held-out planted and null rows.')
    ap.add_argument('submission',type=Path)
    ap.add_argument('--private',type=Path,required=True,help='directory with private_planted.csv and private_noise.csv')
    args=ap.parse_args()
    try: ans=grade_file(args.submission,args.private)
    except (Invalid,ValueError,TypeError,OverflowError,RecursionError,OSError) as e:
        print(json.dumps({'status':'rejected','reason':str(e),'score':0.0},indent=2)); return 2
    print(json.dumps(ans,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
