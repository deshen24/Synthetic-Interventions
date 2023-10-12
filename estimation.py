import os 
import numpy as np
import matplotlib.pyplot as plt

# DGP
from methods import sample_normal_factors, sample_uniform_factors

# matrix
from methods import rowProj, PCR

# intialize random seed
np.random.seed(21248)

# create output directory 
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True) 

#============
# Parameters
#============
T0_list = np.linspace(25, 200, 8)
T0_list = [int(T0) for T0 in T0_list]
T1 = 1 
r = 15
rpre = 10 
mean = 0 
factor_std = 1 
noise_std = 1
low = 0 
high = 1 

# iterations 
num_factor_iters = 50
num_noise_iters = 100
num_iters = num_factor_iters * num_noise_iters

#=============
# Simulations
#=============
# initialize
errs_in_dict = {T0: np.zeros(num_iters) for T0 in T0_list}
errs_out_dict = {T0: np.zeros(num_iters) for T0 in T0_list}

# iterate through different sample sizes
for T0 in T0_list: 
    print("T0 = {}...".format(T0))
    
    # set number of donor units
    Nd = T0
    
    # initialize 
    idx = 0
    
    # iterate through different factor samplings
    for i in range(num_factor_iters): 
    
        # sample unit factors
        V_donors = sample_normal_factors(n=Nd, p=r, mean=mean, std=factor_std, normalize=False)
        w = sample_uniform_factors(n=Nd, p=1, low=low, high=high, normalize=True)
        v_target = (V_donors.T @ w).flatten() 

        # sample measurement-intervention factors
        A = sample_normal_factors(n=T0, p=rpre, mean=mean, std=factor_std, normalize=False)
        B = sample_normal_factors(n=r, p=rpre, mean=mean, std=factor_std, normalize=False)
        U_pre = A @ B.T 
        phi = sample_uniform_factors(n=r, p=1, low=low, high=high, normalize=False).flatten() 
        (P, P_perp) = rowProj(U_pre)
        u_post_in = P @ phi
        u_post_out = P_perp @ phi 

        # construct latent pre-intervention data
        E_ypre_target = U_pre @ v_target
        E_ypre_donors = U_pre @ V_donors.T 
        
        # construct two sets of latent post-intervention data
        E_ypost_donors_in = V_donors @ u_post_in
        theta_in = np.dot(v_target, u_post_in)
        E_ypost_donors_out = V_donors @ u_post_out
        theta_out = np.dot(v_target, u_post_out)
        
        # iterate through different noise samplings
        for j in range(num_noise_iters): 
            
            # sample observations 
            ypre_target = E_ypre_target + np.random.normal(loc=mean, scale=noise_std, size=T0)
            ypre_donors = E_ypre_donors + np.random.normal(loc=mean, scale=noise_std, size=(T0, Nd))
            ypost_donors_in = E_ypost_donors_in + np.random.normal(loc=mean, scale=noise_std, size=Nd)
            ypost_donors_out = E_ypost_donors_out + np.random.normal(loc=mean, scale=noise_std, size=Nd)

            # PCR-variant of SI (learn SINGLE model)
            w_hat = PCR(ypre_donors, ypre_target, k=rpre)

            # point estimate
            theta_hat_in = np.dot(ypost_donors_in, w_hat)
            theta_hat_out = np.dot(ypost_donors_out, w_hat)
            
            # record error
            errs_in_dict[T0][idx] = np.abs(theta_hat_in - theta_in)
            errs_out_dict[T0][idx] = np.abs(theta_hat_out - theta_out)
            idx += 1  
    
print("Done!")
    
#========
# Report
#========
means_in = np.array([errs_in_dict[T0].mean() for T0 in T0_list])
stds_in = np.array([errs_in_dict[T0].std() for T0 in T0_list])
means_out = np.array([errs_out_dict[T0].mean() for T0 in T0_list])
stds_out = np.array([errs_out_dict[T0].std() for T0 in T0_list])

# plot results
fname = os.path.join(output_dir, "consistency")
plt.plot(T0_list, means_out, marker='.', color='darkorange', label='A8 fails')
plt.fill_between(T0_list, means_out-stds_out, means_out+stds_out, alpha=0.3, color='darkorange')
plt.plot(T0_list, means_in, marker='.', color='royalblue', label='A8 holds')
plt.fill_between(T0_list, means_in-stds_in, means_in+stds_in, alpha=0.3, color='dodgerblue')
plt.title('Point estimation')
plt.ylabel('Bias ($|\widehat{\\theta}_n^{(d)} - \\theta_n^{(d)}|$)')
plt.xlabel('Data size ($T_0=N_d$)')
plt.legend(loc='best')
plt.grid(color='lightgrey')
plt.savefig(fname, dpi=400, bbox_inches="tight")
plt.close() 


