import numpy
import pytensor
import pytensor.tensor
from pdb import set_trace

def select_key_or_default(dirc, key, default = None, stack = True, 
                          numPCK = __import__('numpy')):
    dirc = dict(sorted(dirc.items()))
    out = [dirc[name] for name in dirc if key in name]
    
    if len(out) == 0: 
        out = default 
    elif len(out) == 1: 
        out = out[0]
    else: 
        if stack: out = numPCK.stack([i[0] for i in out])
        
    if type(out) is list:         
        try:    
            if stack: out =  numPCK.stack(out)[:,0]
        except:
            pass
       
    return(out)
