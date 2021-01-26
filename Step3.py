import os
os.chdir('/Users/yufei/Documents/2-CMU/DebiasingCvxConstrained/Code/Library')

from Step2 import all_knots

import numpy as np
from math import log

from sklearn.isotonic import IsotonicRegression




#######
# Monotone Cone and Positive Monotone Cone
#######
def solve_omega_mnt(v, v_const, n, Sig_hat, db_idx, pos=False, learning_rate=0.01, stop_criteria=10**-4):
    """
    Solve the optimization in step 3 where K is a monotone cone.
    
    @param v np.array(p, ): the v found in step 2.
    @param v_const int: the number of constant pieces in v.
    @param n int: sample size.
    @param Sig_hat: the sample covariance matrix.
    @param db_idx: the index of the coordinate which will be debiased.
    @param pos bool: True if K is positive mnt cone, otherwise False
    @param learning_rate: step size of the optimization is learning_rate/i.
    @param stop_criteria: stop criteria of the optimization.
    @return omega np.array(p, ): the vector used to debiase the db_idx coordiante.
    """
    p = len(v)
    ej = np.zeros(p)
    ej[db_idx] = 1
    lbd = ( v_const/n * log(np.exp(1)*p/v_const) )**0.5
    
    # Initialize omega
    omega_prev = np.ones(p)
    omega = np.random.normal(size = p)
    
    # Subgradient descent
    i = 0.0 # iteration number
    while sum((omega - omega_prev)**2)**0.5 > stop_criteria:
        i += 1
        
        # calculate subgradient
        if pos:
            proj_pos = proj_posmnt_tan_cone(Sig_hat @ omega - ej, v)
            proj_neg = proj_posmnt_tan_cone(Sig_hat @ omega - ej, v, True)
        else:
            proj_pos = proj_mnt_tan_cone(Sig_hat @ omega - ej, v)
            proj_neg = proj_mnt_tan_cone(Sig_hat @ omega - ej, v, True)
        dot_pos, dot_neg = (Sig_hat @ omega - ej).T @ proj_pos, (Sig_hat @ omega - ej).T @ proj_neg
        
        if max(dot_pos, dot_neg) <= lbd:
            omega_grad = Sig_hat @ omega
        else:
            omega_grad = Sig_hat @ ( proj_pos if dot_pos > dot_neg else proj_neg )
        omega_grad = omega_grad / sum((omega_grad)**2)**0.5
        
        # update omega_prev and omega
        omega_prev = omega
        omega = omega - (learning_rate / i) * omega_grad
    return omega




#######
# LASSO, SLOPE and Square-root SLOPE
#######
def solve_omega_l1_ball(v, n, Sig_hat, db_idx, learning_rate=0.01, stop_criteria=10**-4):
    """
    Solve the optimization in step 3 where K = {x: ||x||_1 <= ||v||_1}.
    
    @param v: the vector whose l1 norm is used to define the convex set K.
    @param n: the sample size.
    @param Sig_hat: the sample covariance matrix.
    @param db_idx: the index of the coordinate which will be debiased.
    @param learning_rate: step size of the optimization is learning_rate/i.
    @param stop_criteria: stop criteria of the optimization.
    
    @return omega np.ndarray (p,): the vector used to debiase the db_idx coordiante.
    """
    p = len(v)
    s = len(np.where(abs(v)>0)[0])
    ej = np.zeros(p)
    ej[db_idx] = 1
    lbd = (s/n * log(p/s))**0.5
    
    # Initialize omega
    omega_prev = np.ones(p)
    omega = np.random.normal(size = p)
    
    # Subgradient descent
    i = 0.0 # iteration number, used to control step size
    while np.sum((omega - omega_prev)**2)**0.5 > stop_criteria:
        i += 1
        if i % 100 == 0: print('100 iterations in step3.')
        
        # calculate subgradient
        proj_pos = proj_l1_tan_cone(Sig_hat @ omega - ej, v)
        proj_neg = proj_l1_neg_tan_cone(Sig_hat @ omega - ej, v)
        dot_pos, dot_neg = (Sig_hat @ omega - ej).T @ proj_pos, (Sig_hat @ omega - ej).T @ proj_neg
        
        if max(dot_pos, dot_neg) <= lbd:
            omega_grad = Sig_hat @ omega
        else:
            omega_grad = Sig_hat @ ( proj_pos if dot_pos > dot_neg else proj_neg )
        omega_grad = omega_grad / np.sum((omega_grad)**2)**0.5
        
        # update omega_prev and omega
        omega_prev = omega
        omega = omega - (learning_rate / i) * omega_grad
    return omega




#######
# Auxiliary Functions
#######
def proj_mnt_tan_cone(u, v, neg=False):
    """
    Project u onto the tangent cone or negative tangent cone of monotone cone at v.
    
    @param u: the vector to calculate projection from.
    @param v: the vector at whom the tangent cone forms.
    
    @return normalized projection of u.
    """
    # Find all knots of v
    knots_list = all_knots(v)

    # Do isotonic regression of u on every constant piece
    for i in range(1, len(knots_list)):
        u_piece = u[knots_list[i-1]:knots_list[i]]
        if neg: iso_order = np.arange(len(u_piece),0, -1)  # The negative of tangent cone consists of mnt decreasing cones
        else: iso_order = np.arange(len(u_piece))
        u[knots_list[i-1]:knots_list[i]] = IsotonicRegression().fit_transform(iso_order, u_piece)
    return u / np.sum(u**2)**0.5
    
    
def proj_posmnt_tan_cone(u, v, neg=False):
    """
    Project u onto the tangent cone or negative tangent cone of positive monotone cone at v.
    
    @param u: the vector to calculate projection from.
    @param v: the vector at whom the tangent cone forms.
    
    @return normalized projection of u.
    """
    # Find all knots of v
    knots_list = all_knots(v)

    # Do isotonic regression of u on every constant piece.
    for i in range(1, len(knots_list)):
        u_piece = u[knots_list[i-1]:knots_list[i]]
        if neg: iso_order = np.arange(len(u_piece),0, -1)  # The negative of tangent cone consists of mnt decreasing cones
        else: iso_order = np.arange(len(u_piece))
        mnt_proj = IsotonicRegression().fit_transform(iso_order, u_piece)
        # if the first constant piece is 0, project to positive monotone cone.
        if i == 1 and u_piece[0] == 0:
            mnt_proj = np.where(mnt_proj > 0, mnt_proj, 0)
        # update u
        u[knots_list[i-1]:knots_list[i]] = mnt_proj
    return u / np.sum(u**2)**0.5


def proj_l1_tan_cone(u, v):
    """
    Project u onto the tangent cone of l1 ball with diameter ||v||_1 at v.
    
    @param u: the vector to calculate projection from.
    @param v: the vector at whom the tangent cone forms.
    
    @return normalized projection of u.
    """
    idx_nonzero = np.where(v != 0)[0]
    idx_zero = np.where(v == 0)[0]

    f_nonzero = lambda x: np.sum((u[idx_nonzero] - x * np.sign(v[idx_nonzero]))**2)
    shrink = lambda x: np.sum((np.sign(u[idx_zero]) * np.clip(abs(u[idx_zero])-x, 0, None))**2)
    f = lambda x: f_nonzero(x) + shrink(x)
    
    # one-dimensional optimization to get t
    t = gss(f,
            0,  # lower bound is 0
            max(abs(u)),  # upper bound is max|u_i|. After t >= max|u_i|, the function will be increasing with t.
            tol=1e-5)
    
    # use t to get the projection
    proj = np.zeros(len(u))
    for i in idx_nonzero:
        proj[i] = t * np.sign(v[i])
    for i in idx_zero:
        proj[i] = u[i] if abs(u[i]) <= t else t * np.sign(u[i])
    
    # moreau's decomposition
    proj = u - proj
    return proj / np.sum(proj**2)**0.5
    
    
def proj_l1_neg_tan_cone(u, v):
    """
    Project u onto the negative tangent cone of l1 ball with diameter ||v||_1
    at v. It's actually the tangent cone of l1 ball with bound ||v||_1 at -v.
    
    @param u: the vector to calculate projection from.
    @param v: the vector at whom the tangent cone forms.
    
    @return normalized projection of u.
    """
    return proj_l1_tan_cone(u, -v)
    
    
# The function to implement golden section search.
# Obtained from https://en.wikipedia.org/wiki/Golden-section_search#Probe_point_selection
gr = (5**0.5 + 1) / 2 # the golden ratio.
def gss(f, a, b, tol=1e-5):
    """Golden section search
    to find the minimum of f on [a,b]
    f: a strictly unimodal function on [a,b]

    Example:
    >>> f = lambda x: (x-2)**2
    >>> x = gss(f, 1, 5)
    >>> print("%.15f" % x)
    2.000009644875678
    """
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    while abs(c - d) > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c

        # We recompute both c and d here to avoid loss of precision which may lead to incorrect results or infinite loop
        c = b - (b - a) / gr
        d = a + (b - a) / gr
    return (b + a) / 2
