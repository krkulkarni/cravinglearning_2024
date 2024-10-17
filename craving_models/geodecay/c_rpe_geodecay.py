import numpy as np
from scipy.stats import binom, multivariate_normal
from scipy.special import logsumexp, expit, logit, softmax
from scipy.stats import norm
from scipy.optimize import minimize

from pyEM.math import norm2beta, norm2alpha, softmax
from tqdm import tqdm

def fit(params, craving_ratings, actions, outcomes, evs, rpes, prior=None, output='npl'):
    ''' 
    Fit the basic craving model to a single subject's data.
        craving_ratings is a np.array with craving ratings for each trial
        outcomes is a np.array with 1 (cue) or 0 (no cue) for each trial
        evs is a np.array with the expected values for each trial
        rpes is a np.array with the prediction errors for each trial
        output is a string that specifies what to return (either 'nll' or 'all')
    '''
    nparams = len(params)
    gamma = norm2alpha(params[0])
    craving_baseline = params[1]
    rpe_weight = params[2]

    if craving_baseline < -10 or craving_baseline > 60:
        return 10000000

    nblocks, ntrials = outcomes.shape
    # replace -1 with NaN in cravinggamma = norm2alpha(params[2])_ratings
    nan_craving_ratings = np.where(craving_ratings == -1, np.nan, craving_ratings)
    # Count number of non NaN values in craving_ratings
    ncravingtrials = np.sum(~np.isnan(nan_craving_ratings))

    # ev          = np.zeros((nblocks, ntrials+1, 2))
    # ch_prob     = np.zeros((nblocks, ntrials,   2))
    # choices_A   = np.zeros((nblocks, ntrials,))
    # pe          = np.zeros((nblocks, ntrials,))

    pred_cravings = np.zeros((nblocks, ntrials)) - 1
    negll  = 0

    for b in range(nblocks): #if nblocks==1, use reversals
        for t in range(ntrials):
            if craving_ratings[b, t] > -1:
                # rpe term
                rpe_term = 0

                # RPE term
                for j in range(0, t+1):
                    rpe_term += rpes[b, j] * gamma**(t-j)
                rpe_term *= rpe_weight

                pred_cravings[b, t] = craving_baseline + rpe_term
        
        resid_sigma = np.std(np.subtract(craving_ratings[b], pred_cravings[b]))
        # get the total negative log likelihood
        negll += -np.sum(norm.logpdf(craving_ratings[b], loc=pred_cravings[b], scale=resid_sigma))
    
    if output == 'npl':
        if prior is not None:  # EM-fit: P(Choices | h) * P(h | O) should be maximised, therefore same as minimizing it with negative sign
            fval = -(-negll + prior['logpdf'](params))

            if any(prior['sigma'] == 0):
                this_mu = prior['mu']
                this_sigma = prior['sigma']
                this_logprior = prior['logpdf'](params)
                print(f'mu: {this_mu}')
                print(f'sigma: {this_sigma}')
                print(f'logpdf: {this_logprior}')
                print(f'fval: {fval}')
            
            if np.isinf(fval):
                fval = 10000000
            return fval
        else: # NLL fit 
            return negll
        
    elif output == 'all':
        subj_dict = {'params'     : [gamma, craving_baseline, rpe_weight],
                     'ev'         : evs, 
                     'actions'    : actions,
                     'outcomes'   : outcomes, 
                     'rpes'       : rpes, 
                     'craving_ratings': craving_ratings,
                     'pred_cravings': pred_cravings,
                     'negll'      : negll,
                     'BIC'        : nparams * np.log(ncravingtrials*nblocks) + 2*negll}
        return subj_dict

# class RPEGeoDecay(CravingPrototype):

#     def __init__(self, decision_model):

#         self.craving_name = 'RPE Geo Decay'
#         self.short_name = 'rpe_geodecay'
#         self.parnames = ['craving_baseline', 'rpe_weight', 'gamma', 'craving_sd']
#         self.partransform = ['1*', '1*', 'expit', 'np.exp']
#         self.npars = len(self.parnames)

#         super().__init__(decision_model)

#     def llfunc(self, params, *args):
#         craving_baseline, rpe_weight, gamma, craving_sd = params
#         craving_ratings, q_vals, rpe_vals, actions, rewards, map_prior, return_type = args

#         pred_cravings = []
#         true_cravings = []

#         if not map_prior is False:
#             prior_loglik = multivariate_normal.logpdf(
#                 [craving_baseline, rpe_weight, gamma, craving_sd], 
#                 map_prior[:, 0], map_prior[:, 1]
#             )
#         else:
#             prior_loglik = 0

#         if np.abs(craving_baseline) > 100 or np.abs(rpe_weight) > 10 or np.abs(craving_sd) > 10:
#             return 10000000

#         craving_sd = np.exp(craving_sd)
#         gamma = expit(gamma)

#         # Iterate over all trials
#         for i in range(len(craving_ratings)):
#             # Only calculate LL for trials where craving was reported
#             if (craving_ratings[i] > -1):

#                 # outcome term
#                 rpe_term = 0

#                 # EV term
#                 for j in range(0, i+1):
#                     rpe_term += rpe_vals[j] * gamma**(i-j)
#                 rpe_term *= rpe_weight

#                 pred_cravings.append(craving_baseline + rpe_term)
#                 true_cravings.append(craving_ratings[i])

#         # Return the negative summed log likelihood
#         if return_type == 'll':
#             return -prior_loglik - np.sum(norm.logpdf(true_cravings, loc=pred_cravings, scale=craving_sd))
#         elif return_type == 'pred':
#             return np.array(pred_cravings), np.array(true_cravings), -prior_loglik-np.sum(norm.logpdf(true_cravings, loc=pred_cravings, scale=craving_sd))
        