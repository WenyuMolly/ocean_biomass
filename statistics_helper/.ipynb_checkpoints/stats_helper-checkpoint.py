import numpy as np
from scipy.stats import gmean

def mul_CI_SD_SE(vals):
    # Define a function to calculate the multiplicative 95% confidence interval of a list of measurements
    vals = np.stack(vals)
    sem = 10**(np.log10(vals).std()/np.sqrt(len(vals))*1.96)
    sd = 10**(np.log10(vals).std()*1.96)
    return gmean([sem,sd])


'''
def geo_CI_calc(estimates):
    """ 
    This function calculates the 95% multiplicative confidence interval of the input
    
    Input: estimates: numpy array of values
    Output: 95% multiplivative condifence inverval of the geometric mean of the values in the input
    """
    mul_CI_sem = 10**(np.log10(estimates).std(ddof=1)/np.sqrt(len(estimates))*1.96)
    mul_CI_sd = 10**(np.log10(estimates).std(ddof=1)*1.96)
    return gmean([mul_CI_sem,mul_CI_sd])




def CI_prod_prop(mul_CIs):
    mul_CIs = np.stack(mul_CIs)
    # Define a function to propagate uncertainties through a product
    return 10**np.sqrt((np.log10(mul_CIs)**2).sum())

def CI_sum_prop(estimates, mul_CIs):
    """
    This function calculates the 95% confidence interval of a sum of two estimates. 
    We assume these estimates are distributed lognormally with 95% confidence interval provided as input
    Input:
        estimates: numpy array of the estimates to sum over
        mul_CIs: numpy array containing the 95% confidence interval for each estimate in the argument estimates
    Output: 95% multiplivative condifence inverval of the sum of the estimates
    """
    sample_size = 100000
    data = np.zeros([0,sample_size])
    
    # Iterate over the estimates 
    for ind, estimate in enumerate(estimates):
        # For each estimate, sample 1000 samples from a lognormal distribution with a mean of log(estimate) and std of log(95_CI)/1.96
        # This generates an array with N rows and 1000 columns, where N is the number of estimates in the argument estimates
        sample = np.random.lognormal(mean = np.log(estimate), sigma = np.log(mul_CIs[ind])/1.96,size=sample_size).reshape([1,-1])
        data = np.vstack((data,sample))

    # Sum over the N estimates to generate a distribution of sums
    data_sum = data.sum(axis=0)    

    # Calculate the multiplicative value of the 97.5 percentile relative to the mean of the distribution
    upper_CI = np.percentile(data_sum, 97.5)/np.mean(data_sum)

    # Calculate the multiplicative value of the mean of the distribution relative to the 2.5 percentile
    lower_CI = np.mean(data_sum)/np.percentile(data_sum, 2.5)

    # Return the mean of the upper and lower multiplicative values
    return np.mean([upper_CI,lower_CI])
    
'''