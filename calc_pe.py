import numpy as np
import sympy as sp
UR5E_DH_A     = [0.0, -0.425, -0.3922, 0.0, 0.0, 0.0]
UR5E_DH_D     = [0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996]
UR5E_DH_ALPHA = [np.pi/2, 0.0, 0.0, np.pi/2, -np.pi/2, 0.0]

q1, q2, q3, q4, q5, q6 = sp.symbols('q1 q2 q3 q4 q5 q6')

def dh(a, alpha, d, theta):
    ct, st = sp.cos(theta), sp.sin(theta)
    ca, sa = sp.cos(alpha), sp.sin(alpha)
    return sp.Matrix([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d   ],
        [0,   0,      0,     1   ]])

T1 = dh(UR5E_DH_A[0], UR5E_DH_ALPHA[0], UR5E_DH_D[0], q1)
T2 = dh(UR5E_DH_A[1], UR5E_DH_ALPHA[1], UR5E_DH_D[1], q2)
T3 = dh(UR5E_DH_A[2], UR5E_DH_ALPHA[2], UR5E_DH_D[2], q3)
T4 = dh(UR5E_DH_A[3], UR5E_DH_ALPHA[3], UR5E_DH_D[3], q4)
T5= dh(UR5E_DH_A[4], UR5E_DH_ALPHA[4], UR5E_DH_D[4], q5)
T6 = dh(UR5E_DH_A[5], UR5E_DH_ALPHA[5], UR5E_DH_D[5], q6)

P_e = (T1 * T2 * T3 * T4 * T5 * T6).applyfunc(sp.simplify)

P_ee = P_e[:3, 3]

P_eee = P_ee.jacobian([q1, q2, q3, q4, q5, q6])

print(P_eee)