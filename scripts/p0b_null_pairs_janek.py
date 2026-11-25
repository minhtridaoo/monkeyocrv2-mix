"""P0b null: per-image oracle gain between near-identical model pairs on A (luck/threshold floor)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from mixab import reselect_split_janek as rs 
from p0b_per_image_oracle import best_choice
R=Path('H:/My Drive/CS431_MixA/runs/merge_audit_v1_A100-SXM4-40GB')
L=lambda n: rs.load_counts(R/f'{n}_source_counts.npz')
d={n:L(n) for n in ['B-Clean','merge_a1_2','merge_a3_4','B1']}
keys=[rs.setting_key(s) for s in d['B1']['settings']]
split=np.array([rs.split_of(s) for s in d['B1']['source_ids']]); ix=np.where(split=='A')[0]
def pair(n1,k1,n2,k2):
    out=[]
    for c in (0,1):
        X=np.stack([d[n1]['counts'][ix,c,keys.index(k1)],d[n2]['counts'][ix,c,keys.index(k2)]],1).astype(float)
        M,D,G=X[:,:,0],X[:,:,2],X[:,0,1]
        h1=2*M[:,0].sum()/(G.sum()+D[:,0].sum()); h2=2*M[:,1].sum()/(G.sum()+D[:,1].sum())
        _,h=best_choice(M,G,D); out.append((h-max(h1,h2))*100)
    return [round(x,2) for x in out]
print('null a1_2 vs a3_4 (eps/0.50), gain clean,lr2 pp:',pair('merge_a1_2','epsilon_005/score_0.50','merge_a3_4','epsilon_005/score_0.50'))
print('null a3_4 vs B1 (eps/0.50):',pair('merge_a3_4','epsilon_005/score_0.50','B1','epsilon_005/score_0.50'))
print('null B1 two settings eps0.50 vs eps0.60:',pair('B1','epsilon_005/score_0.50','B1','epsilon_005/score_0.60'))
print('real B-Clean vs B1:',pair('B-Clean','mask_040/score_0.60','B1','epsilon_005/score_0.50'))
