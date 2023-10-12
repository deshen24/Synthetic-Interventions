import numpy as np 

#=====
# DGP
#=====

# sample latent factors from a normal distribution 
def sample_normal_factors(n=1, p=1, mean=0, std=1, normalize=False): 
    x = np.random.normal(loc=mean, scale=std, size=(n, p))
    if normalize: 
        x /= np.linalg.norm(x)
    return x

# sample latent factors from a uniform distribution 
def sample_uniform_factors(n=1, p=1, low=0, high=1, normalize=False): 
    x = np.random.uniform(low=low, high=high, size=(n, p))
    if normalize: 
        x /= np.linalg.norm(x)
    return x

#===================
# Matrix operations
#===================

# construct orthogonal projection matrix onto rowspan 
def rowProj(X): 
    P = np.linalg.pinv(X) @ X
    return (P, np.eye(X.shape[1]) - P) 

# principal component regression (PCR)
def PCR(X, y, k): 
    (u, s, v) = np.linalg.svd(X)
    u_k = u[:, :k]
    v_k = v[:k, :]
    s_k = s[:k]
    return ((v_k.T / s_k) @ u_k.T) @ y

# singular value thresholding (SVT)
def SVT(X, k): 
    (u, s, v) = np.linalg.svd(X)
    u_k = u[:, :k]
    v_k = v[:k, :]
    s_k = s[:k]
    return (u_k * s_k) @ v_k 

#===========
# INFERENCE
#===========

# confidence interval
def confidenceInterval(y, X, w, theta_hat, T1=1, z=1.96): 
    sigma = np.linalg.norm(y - (X@w), 2) / np.sqrt(X.shape[0])
    w_norm = np.linalg.norm(w, 2)
    delta = (z*sigma*w_norm) / T1
    return (theta_hat-delta, theta_hat+delta)

# interval length
def interval_len(ub, lb):
    return ub-lb

# theoretical variance
def theoreticalVariance(std, w, X): 
    (P, _) = rowProj(X)
    w_min_norm = np.linalg.norm(P@w, 2) 
    return (std * w_min_norm) ** 2