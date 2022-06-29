import numpy as np

DIV = 1/np.sqrt(2)

def hellinger_distance(p, q):
    return DIV * np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q))**2))
