import numpy as np

from sklearn.isotonic import IsotonicRegression

from cvxopt import matrix, solvers
solvers.options['show_progress'] = False




#######
# Monotone Cone and Positive Monotone Cone
#######
def solve_beta_mnt(X, Y, pos=False, learning_rate=0.01, stop_criteria=10**-4):
    """
    Solve the beta for monotone cone and positive monotone cone.
    
    @param X np.array:
    @param Y np.array:
    @param pos bool: True if K is positive mnt cone, otherwise False
    @param learning_rate float: the step size is learning_rate/i
    @param stop_criteria float: the stop criteria
    @return np.array: the coefficient estimation by constrained lasso.
    
    Test:
    n, p = 100, 250
    beta = np.array([0]*int(0.7*p) + [1]*(p-int(0.7*p)))
		X = np.random.normal(size = (n,p)) @ Sigma_sqrt
		Y = X @ beta + np.random.normal(0, noise_sd, n)

		beta_hat = mnt_reg(X, Y)
		print(beta_hat)
    """
    n = len(Y)
    p = X.shape[1]
    iso_order = np.arange(p)
    
    # initialize
    beta_prev = np.ones(p)
    beta = np.random.normal(size = X.shape[1])
    
    # gradient descent
    i = 0.0  # iteration number
    while sum((beta-beta_prev)**2)**0.5 > stop_criteria:
        i += 1
#         print(sum((beta-beta_prev)**2)**0.5)  # used for debug
        
        # calculate gradient
        beta_grad = -2/n * (X.T@Y - X.T@X@beta)
        # update beta_prev
        beta_prev = beta
        # update beta with projection
        beta = beta - (1/i) * learning_rate * beta_grad
        beta = IsotonicRegression().fit_transform(iso_order, beta)
        # if pos == True, assign zero to negative coordinates
        if pos: beta = np.where(beta > 0, beta, 0)
#         print(sum((beta-beta_prev)**2)**0.5) # used for testing
    return beta




#######
# LASSO
#######
def solve_beta_lasso(X, Y, t):
    """
    Solve the constrained lasso with l1 norm bound t.
    
    @param X np.array:
    @param Y np.array:
    @param t float: l1 norm bound
    @return np.array: the coefficient estimation by constrained lasso.
    """
    p = X.shape[1]
    cov_mat = X.T @ X
    cov_mat = np.concatenate((cov_mat, cov_mat), 0)
    xy = X.T @ Y * (-2.0)
    
    # QP to solve concatenated beta
    P = matrix(np.concatenate((cov_mat, cov_mat), 1), tc='d')
    q = matrix(np.concatenate((xy, xy), 0), tc='d')
    G = matrix(np.diag(np.concatenate((-1.0*np.ones(p), np.ones(p)), 0)), tc='d')
    h = matrix(np.zeros(2*p), tc='d')
    A = matrix(np.concatenate((np.ones(p), -1.0*np.ones(p)), 0).reshape((1,2*p)), tc='d')
    b = matrix(t, tc='d')
    
    # Get the solution of QP
    beta_bundle = np.array(solvers.qp(P,q,G,h,A,b)['x'])
    
    # Reconstruct beta, assign zero to the very small coordinates
    beta = beta_bundle[:p] + beta_bundle[p:]
#     print(sum(abs(beta)))
    beta = np.where(beta > 10**-4, beta, 0)
    return np.squeeze(beta)




#######
# SLOPE
#######
def solve_beta_slope(X, Y, lbd_vec, h=0.1, lr=5.0):
    """
    Solve the SLOPE by proximal gradient descent.
    
    @param X np.array: the data matrix.
    @param Y np.array: the response vector.
		@param lbd_vec: the tuning param according to the 'slope meets lasso' paper
		@param h float: the step size of optimization. It needs to be small enough (h/n < 2/||X|| according to Bogdan et al.)
		@param lr float: the learning rate of optimization. Use a large value to avoid overfit.
    @return np.array: the coefficient got by SLOPE.
    """
    n, p = X.shape[0], X.shape[1]
    
#    i = 0
    beta_prev = np.zeros(p)
    beta_new = np.ones(p)
    while abs(obj_slope(X, Y, lbd_vec, beta_prev)-obj_slope(X, Y, lbd_vec, beta_new)) > lr:
        beta_prev = beta_new
        beta_new = prox_slope(beta_new - (h/n) * (X.T @ (X @ beta_new - Y)), h/n, lbd_vec)
        
#        i += 1
#        if i % 2 == 0:
#            print(i)
#            print("prev value: ", obj_slope(X, Y, lbd_vec, beta_prev))
#            print("new value: ", obj_slope(X, Y, lbd_vec, beta_new))
#            print(sum(abs(beta_new)))
#            print(beta_new)
    return beta_new




#######
# Square-root SLOPE
#######
def solve_beta_sqrt_slope(X, Y, lbd_vec, h=2.0, lr=5.0):
    """
    Solve the square-root SLOPE.

    @param X np.array: the data matrix.
    @param Y np.array: the response vector.
		@param lbd_vec: the tuning param according to Derumigny's paper
		@param h float: the step size of optimization is h/i.
		@param lr float: the learning rate of optimization. Use a large value to avoid overfit.
    @return np.array: the coefficient got by square-root SLOPE.
    """
    p = X.shape[1]

    sigma_prev, sigma_new = np.var(Y)**0.5, np.var(Y)**0.5
    beta_prev, beta_new = np.zeros(p), solve_beta_slope(X, Y, sigma_new*lbd_vec)

    i = 1.0
    while abs(obj_sqrt_slope(X, Y, lbd_vec, beta_prev, sigma_prev) - obj_sqrt_slope(X, Y, lbd_vec, beta_new, sigma_new)) > lr:
        sigma_prev, beta_prev = sigma_new, beta_new
        sigma_new = sigma_new - (h/i) * ( 1 - np.var(Y - X@beta_new)/sigma_new**2 )
        beta_new = solve_beta_slope(X, Y, sigma_new*lbd_vec)
        i += 1
        
#        if i % 100 == 0: print('step1: i=', i)
#        print("i=", i)
#        print('sigma_prev, sigma_new: ', sigma_prev, sigma_new)
#        print('obj value:', obj_sqrt_slope(X, Y, lbd_vec, beta_new, sigma_new))
#        print('difference: ', abs(obj_sqrt_slope(X, Y, lbd_vec, beta_prev, sigma_prev) - obj_sqrt_slope(X, Y, lbd_vec, beta_new, sigma_new)))
    return beta_new
    



#######
# Auxiliary Functions
#######
def prox_slope(x, h, lbd):
    """
    Get the prox mapping in each step of the SLOPE solver.
    
    @param x np.array (p, ): the vector to be mapped.
    @param h float: the step size of proximal method.
    @param lbd np.array (p, ): the vector of lambda.
    @return np.array (p, ): the prox mapping of x.
    """
    # reorder the lambda to make it coincide with the order of x
    sort_idx = np.argsort(abs(x))
    rank_x = np.arange(len(x))[np.argsort(sort_idx)]
    return np.sign(x) * np.clip(abs(x) - lbd[rank_x] * h, 0, None)


def obj_slope(X, Y, lbd, beta):
    """
    Evaluate the objective function in the optimization of SLOPE.
    
    @param X np.array: the data matrix.
    @param Y np.array: the response vector.
    @param lbd np.array: the constraint parameter.
    @param beta np.array: the coefficient.
    """
    n = X.shape[0]
    return np.sum((Y - X@beta)**2)/n + np.sum(lbd * np.sort(abs(beta))[::-1])


def obj_sqrt_slope(X, Y, lbd, beta, sigma):
    """
    Evaluate the objective function in the optimization of square-root SLOPE.
    
    @param X np.array: the data matrix.
    @param Y np.array: the response vector.
    @param lbd np.array: the constraint parameter.
    @param beta np.array: the coefficient.
    @param sigma float: the standard error of noise.
    """
    n = X.shape[0]
    return sigma + np.sum((Y - X@beta)**2) / (2 * n * sigma) + np.sum(sigma * lbd * np.sort(abs(beta))[::-1])
