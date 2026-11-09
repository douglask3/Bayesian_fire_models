import numpy
import pytensor
import pytensor.tensor
from pdb import set_trace

def contains_with_max_extra(text, substring, max_extra=2):
    """
    Returns True if substring is contained in text and
    text has max_extra or fewer additional characters.
    """
    return substring in text and (len(text) - len(substring)) <= max_extra

def select_key_or_default(dirc, key, default = None, stack = True, 
                          numPCK = __import__('numpy')):
    dirc = dict(sorted(dirc.items()))
    out = [dirc[name] for name in dirc if contains_with_max_extra(name, key)]
    #set_trace()
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
