# run with: uv run --no-project --with matplotlib --with numpy python make_plots_3d.py
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
X=np.loadtxt('example_X.csv',delimiter=',')
n,d=X.shape; R=np.random.default_rng(0).random((n,d))
DC,NC='#1f6feb','#6e7781'
def pcs(A):
    A=A-A.mean(0); return A@np.linalg.svd(A,full_matrices=False)[2][:3].T
def panel(fig,pos,P,color,title,labels):
    ax=fig.add_subplot(*pos,projection='3d'); ax.scatter(*P.T,s=2,color=color,alpha=.35)
    ax.set(xlabel=labels[0],ylabel=labels[1],zlabel=labels[2]); ax.set_title(title,fontsize=10); ax.view_init(22,35)
    return ax
# 7: top-3 PCs, data next to noise
fig=plt.figure(figsize=(11,5.2))
panel(fig,(1,2,1),pcs(X),DC,'example_X',('PC1','PC2','PC3'))
panel(fig,(1,2,2),pcs(R),NC,'uniform noise, same shape',('PC1','PC2','PC3'))
fig.suptitle('Top-3 PCA projection in 3D: the data is the same featureless blob as noise')
fig.tight_layout(); fig.savefig('plots/07_pca_3d_data_vs_noise.png',dpi=140); plt.close()
# 8: same data projection from four viewing angles
P=pcs(X); fig=plt.figure(figsize=(10,9))
for k,(el,az) in enumerate([(22,35),(22,125),(60,35),(0,0)]):
    ax=panel(fig,(2,2,k+1),P,DC,f'elevation {el}°, azimuth {az}°',('PC1','PC2','PC3')); ax.view_init(el,az)
fig.suptitle('example_X, top-3 PCA projection from four angles: no shape from any side')
fig.tight_layout(); fig.savefig('plots/08_pca_3d_four_views.png',dpi=140); plt.close()
# 9: raw column triples (the highest-scoring pairs plus a third column)
fig=plt.figure(figsize=(11,5.2))
for k,t in enumerate([(37,42,6),(6,13,10)]):
    panel(fig,(1,2,k+1),X[:,t],DC,f'columns {t[0]}, {t[1]}, {t[2]}',[f'col {c}' for c in t])
fig.suptitle('Raw column triples built from the highest-scoring pairs: evenly filled cubes')
fig.tight_layout(); fig.savefig('plots/09_column_triples_3d.png',dpi=140); plt.close()
