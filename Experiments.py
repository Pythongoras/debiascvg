import numpy as np




#######
# Monotone Cone and Positive Monotone Cone
#######
"""
Implement the experiments for monotone cone or positive monotone cone.

@param N int: number of repetitions
@param n int: sample size
@param p int: dimension
@param Sigma_sqrt np.array(p, p): square root of population cov matrix
@param noise_sd float: the standard deviation of noise
@param debias_index int: the index of coordinate to debias
@param pos bool: True if K is positive mnt cone, otherwise False
@param step1_func function: the function to solve beta_hat in step1
@param step2_func function: the function to solve v in step2
@param step3_func function: the function to solve omega in step3
"""
def exp_mnt_cone(N,
                 n,
                 p,
                 Sigma_sqrt,
                 noise_sd,
                 debias_index,
                 pos,
                 step1_func,
                 step2_func,
                 step3_func):
    z = []
    z_biased = []
    
    for i in range(N):
        print("iter:", i)
    
        # generate data
        if pos: beta = np.array([0.0]*int(0.7*p) + [1.0]*(p-int(0.7*p)))
        else: beta = np.array([-1.0]*int(0.7*p) + [1.0]*(p-int(0.7*p)))
        X = np.random.normal(size = (n,p)) @ Sigma_sqrt
        Y = X @ beta + np.random.normal(0, noise_sd, n)
    
        # sample split
        X1, X2 = X[:n//2], X[n//2:]
        Y1, Y2 = Y[:n//2], Y[n//2:]
        Sig_hat = (X2.T @ X2) / n
    
        # step 1: compute beta_hat
        beta_hat = step1_func(X1, Y1, pos)
        print( "The L2 error: ", sum((beta_hat-beta)**2)**0.5 )
        
        # step 2: get v
        v, v_const = step2_func(beta_hat, n)
        print( "The L1 norm of v: ", sum(abs(v)) )
    
        # step 3: get omega and debiase beta_hat
        omega = step3_func(v, v_const, n, Sig_hat, debias_index, pos)
        beta_d = v[debias_index] + 1.0/n * omega.T@X2.T@(Y2-X2@v)
    
        # standardize the debiased beta, append to z
        z_var = omega.T@Sig_hat@omega * np.var(Y1-X1@beta_hat)
        z.append( (beta[debias_index]-beta_d) / (n*z_var)**0.5 )
    
        # append the biased beta to z_biased
        z_biased.append(beta[debias_index]-beta_hat[debias_index])
        
    z_biased = np.array(z_biased) / (n*np.var(z_biased))**0.5
    return z, z_biased




#######
# LASSO
#######
"""
Implement the experiments of LASSO.

@param N int: number of repetitions
@param n int: sample size
@param p int: dimension
@param Sigma_sqrt np.array(p, p): square root of population cov matrix
@param noise_sd float: the standard deviation of noise
@param cardi float <= 1: proportion of non-zero coordinates
@param l1_bound float: use-defined upper bound of l1-norm of the coefficient
@param debias_index int: the index of coordinate to debias
@param step1_func function: the function to solve beta_hat in step1
@param step2_func function: the function to solve v in step2
@param step3_func function: the function to solve omega in step3
"""
def exp_lasso(N,
              n,
              p,
              Sigma_sqrt,
              noise_sd,
              cardi,
              l1_bound,
              debias_index,
              step1_func,
              step2_func,
              step3_func):
    z = []
    z_biased = []
    
    for i in range(N):
        print("iter:", i)
    
        # generate data
        beta = np.array([0]*int(p-int(cardi*p)) + [1]*int(cardi*p))
        X = np.random.normal(size = (n,p)) @ Sigma_sqrt
        Y = X @ beta + np.random.normal(0, noise_sd, n)
    
        # sample split
        X1, X2 = X[:n//2], X[n//2:]
        Y1, Y2 = Y[:n//2], Y[n//2:]
        Sig_hat = (X2.T @ X2) / n
    
        # step 1: compute beta_hat
        beta_hat = step1_func(X1, Y1, l1_bound)
        print( "The L2 error: ", sum((beta_hat-beta)**2)**0.5 )
    
        # step 2: get v
        v = step2_func(beta_hat, l1_bound, n)
        print( "The L1 norm of v: ", sum(abs(v)) )
    
        # step 3: get omega and debiase beta_hat
        omega = step3_func(v, n, Sig_hat, debias_index)
        beta_d = v[debias_index] + 1.0/n * omega.T@X2.T@(Y2-X2@v)
    
        # standardize the debiased beta, append to z
        z_var = omega.T@Sig_hat@omega * np.var(Y1-X1@beta_hat)
        z.append( (beta[debias_index]-beta_d) / (n*z_var)**0.5 )
    
        # append the biased beta to z_biased
        z_biased.append(beta[debias_index]-beta_hat[debias_index])
        
    z_biased = np.array(z_biased) / (n*np.var(z_biased))**0.5
    return z, z_biased




#######
# SLOPE and Square-root SLOPE
#######
"""
Implement the experiments of SLOPE and square-root SLOPE.

@param N int: number of repetitions
@param n int: sample size
@param p int: dimension
@param Sigma_sqrt np.array(p, p): square root of population cov matrix
@param noise_sd float: the standard deviation of noise
@param cardi float <= 1: proportion of non-zero coordinates
@param su int<p: use-defined upper bound of number of non-zero coordinates
@param debias_index int: the index of coordinate to debias
@param lbd_vec np.array(p, ): the decreasing constraint parameter vector
@param C float: the constant in upper bound
@param step1_func function: the function to solve beta_hat in step1
@param step2_func function: the function to solve v in step2
@param step3_func function: the function to solve omega in step3
"""
def exp_slopes(N,
               n,
               p,
               Sigma_sqrt,
               noise_sd,
               cardi,
               su,
               debias_index,
               lbd_vec,
               C,
               step1_func,
               step2_func,
               step3_func):
    z = []
    z_biased = []
    
    for i in range(N):
        print("iter:", i)
    
        # generate data
        beta = np.array( [0]*int(p-int(cardi*p)) + list(np.arange(1, int(cardi*p)+1, 1)) )
        X = np.random.normal(size = (n,p)) @ Sigma_sqrt
        Y = X @ beta + np.random.normal(0, noise_sd, n)
    
        # sample split
        X1, X2 = X[:n//2], X[n//2:]
        Y1, Y2 = Y[:n//2], Y[n//2:]
        Sig_hat = (X2.T @ X2) / n
    
        # step 1: compute beta_hat
        beta_hat = step1_func(X1, Y1, lbd_vec)
        print( "The L2 error: ", sum((beta_hat-beta)**2)**0.5 )
    
        # step 2: get v
        v = step2_func(beta_hat, n, C, su)
        print( "The L1 norm of v: ", sum(abs(v)) )
    
        # step 3: get omega and debiase beta_hat
        omega = step3_func(v, n, Sig_hat, debias_index)
        beta_d = v[debias_index] + 1.0/n * omega.T@X2.T@(Y2-X2@v)
    
        # standardize the debiased beta, append to z
        z_var = omega.T@Sig_hat@omega * np.var(Y1-X1@beta_hat)
        z.append( (beta[debias_index]-beta_d) / (n*z_var)**0.5 )
    
        # append the biased beta to z_biased
        z_biased.append(beta[debias_index]-beta_hat[debias_index])
        
    z_biased = np.array(z_biased) / (n*np.var(z_biased))**0.5
    return z, z_biased
