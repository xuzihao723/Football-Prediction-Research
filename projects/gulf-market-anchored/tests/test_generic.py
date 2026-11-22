import numpy as np
from src.models.generic_ml import smart_vote

def test_smart_vote_training_weights():
    p={'a':np.array([[.2,.3,.5]]),'b':np.array([[.3,.3,.4]]),'c':np.array([[.4,.3,.3]]),'d':np.array([[.5,.2,.3]])}
    out,meta=smart_vote(p,{'a':.7,'b':.6,'c':.5,'d':.4},3)
    assert meta['members']==['a','b','c']; assert abs(sum(meta['weights'])-1)<1e-12; assert out.shape==(1,3)
