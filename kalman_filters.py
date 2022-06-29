import sympy as sp
from ex4_utils import *
from sympy.interactive.printing import init_printing
init_printing(use_unicode=False, wrap_line=False)


class KalmanFilterParams():
    def __init__(self, t=1, r=0.1, q=1):
        self.t = t
        self.r = r
        self.q = q


def calc(Fi, Q, R, params):
    T, q, r = sp.symbols('T q r')
    Fi = np.array(Fi.subs({T: params.t}), dtype=np.float32)
    Q = np.array(Q.subs({T: params.t, q: params.q, r: params.r}), dtype=np.float32)
    R = np.array(R.subs({r: params.r}), dtype=np.float32)
    return Fi, Q, R


def get_RW_model(params):
    T, q, r = sp.symbols('T q r')
    Fi = sp.Matrix([[1, T], [0, 1]])
    L = sp.Matrix([[1, 0], [0, 1]])
    Q = sp.integrate((Fi * L) * q * (Fi * L).T, (T, 0, T))
    R = sp.Matrix([[r, 0], [0, r]])
    H = np.array([[1, 0], [0, 1]])
    Fi, Q, R = calc(Fi, Q, R, params)
    return Fi, Q, R, H


def get_NCV_model(params):
    T, q, r = sp.symbols('T q r')
    Fi = sp.Matrix([[1, 0, T, 0], [0, 1, 0, T], [0, 0, 1, 0], [0, 0, 0, 1]])
    L = sp.Matrix([[0, 0], [0, 0], [1, 0], [0, 1]])
    Q = sp.integrate((Fi * L) * q * (Fi * L).T, (T, 0, T))
    R = sp.Matrix([[r, 0], [0, r]])
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    Fi, Q, R = calc(Fi, Q, R, params)
    return Fi, Q, R, H


def get_NCA_model(params):
    T, q, r = sp.symbols('T q r')
    Fi = sp.Matrix([[1, 0, 0, 0, T, 0], [0, 1, 0, 0, 0, T], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
    L = sp.Matrix([[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 1]])
    Q = sp.integrate((Fi * L) * q * (Fi * L).T, (T, 0, T))
    R = sp.Matrix([[r, 0], [0, r]])
    H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
    Fi, Q, R = calc(Fi, Q, R, params)
    return Fi, Q, R, H


def initialize(x, y, fi):
    sx = np.zeros((x.size, 1), dtype=np.float32).flatten()
    sy = np.zeros((y.size, 1), dtype=np.float32).flatten()
    sx[0] = x[0]
    sy[0] = y[0]
    state = np.zeros((fi.shape[0], 1), dtype=np.float32).flatten()
    state[0] = x[0]
    state[1] = y[0]
    covariance = np.eye(fi.shape[0], dtype=np.float32)
    return state, sx, sy, covariance
