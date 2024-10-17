import numpy as np
from scipy.stats import binom, multivariate_normal
from scipy.special import expit
from scipy.optimize import minimize
# from decision_models.em_prototype import EMPrototype
from pyEM.math import norm2beta, norm2alpha, softmax
from tqdm import tqdm

def fit(params, choices, rewards, craving_ratings, prior=None, output='npl'):
    ''' 
    Fit the basic RW model to a single subject's data.
        choices is a np.array with 0 (left) or 1 (right) for each trial
        rewards is a np.array with 1 (cue) or 0 (no cue) for each trial
        output is a string that specifies what to return (either 'nll' or 'all')
    '''
    nparams = len(params)
    beta = norm2beta(params[0])
    lr_pos   = norm2alpha(params[1])
    lr_neg   = norm2alpha(params[2])

    # make sure params are in range
    this_alpha_bounds = [0, 1]
    if lr_pos < min(this_alpha_bounds) or lr_pos > max(this_alpha_bounds):
        # print(f'lr = {i_alpha:.3f} not in range')
        return 10000000
    if lr_neg < min(this_alpha_bounds) or lr_neg > max(this_alpha_bounds):
        # print(f'lr = {i_alpha:.3f} not in range')
        return 10000000
    this_beta_bounds = [0.00001, 10]
    if beta < min(this_beta_bounds) or beta > max(this_beta_bounds):
        # print(f'beta = {beta:.3f} not in range')
        return 10000000

    nblocks, ntrials = rewards.shape

    ev          = np.zeros((nblocks, ntrials+1, 2))
    ch_prob     = np.zeros((nblocks, ntrials,   2))
    choices_A   = np.zeros((nblocks, ntrials,))
    pe          = np.zeros((nblocks, ntrials,))
    choice_nll  = 0

    for b in range(nblocks): #if nblocks==1, use reversals
        for t in range(ntrials):
            if t == 0:
                ev[b, t,:]    = [.5, .5]

            # get choice index
            if choices[b, t] == 0:
                c = 0
                choices_A[b, t] = 1
            else:
                c = 1
                choices_A[b, t] = 0

            # calculate choice probability
            ch_prob[b, t,:] = softmax(ev[b, t, :], beta)
            
            # calculate PE
            pe[b, t] = rewards[b, t] - ev[b, t, c]

            # update EV
            ev[b, t+1, :] = ev[b, t, :].copy()
            if pe[b, t] >= 0:
                ev[b, t+1, c] = ev[b, t, c] + (lr_pos * pe[b, t])
            else:
                ev[b, t+1, c] = ev[b, t, c] + (lr_neg * pe[b, t])
            
            # add to sum of choice nll for the block
            choice_nll += -np.log(ch_prob[b, t, c])
        
    # get the total negative log likelihood
    negll = choice_nll
    
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
        subj_dict = {'params'     : [beta, lr_pos, lr_neg],
                     'ev'         : ev, 
                     'ch_prob'    : ch_prob, 
                     'choices'    : choices, 
                     'choices_A'  : choices_A, 
                     'rewards'    : rewards, 
                     'pe'         : pe, 
                     'negll'      : negll,
                     'BIC'        : nparams * np.log(ntrials*nblocks) + 2*negll}
        return subj_dict
def simulate(params, block_type, raw_craving_ratings, nblocks=1, ntrials=60):
    """
    Simulate the basic RW model.

    Args:
        `params` is a np.array of shape (nsubjects, nparams)
        `nblocks` is the number of blocks to simulate
        `ntrials` is the number of trials per block
    
    Returns:
        `simulated_dict` is a dictionary with the simulated data with the following keys:
            - `ev` is a np.array of shape (nsubjects, nblocks, ntrials+1, 2)
            - `ch_prob` is a np.array of shape (nsubjects, nblocks, ntrials, 2)
            - `choices` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `choices_A` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `rewards` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `pe` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `choice_nll` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `params` is a np.array of the parameters used to simulate the data
                - `beta` is the softmax inverse temperature
                - `lr` is the learning rate
    """

    nsubjects   = params.shape[0]
    ev          = np.zeros((nsubjects, nblocks, ntrials+1, 2))
    ch_prob     = np.zeros((nsubjects, nblocks, ntrials,   2))
    choices     = np.empty((nsubjects, nblocks, ntrials,), dtype='object')
    choices_A   = np.zeros((nsubjects, nblocks, ntrials,))
    rewards     = np.zeros((nsubjects, nblocks, ntrials,))
    craving_ratings = np.zeros((nsubjects, nblocks, ntrials,))
    pe          = np.zeros((nsubjects, nblocks, ntrials,))
    choice_nll  = np.zeros((nsubjects, nblocks, ntrials,))

    subj_dict = {}

    if block_type == 'money':
        # reward vec is the reward probabilities for each trial (60 total trials)
        reward_vec = np.array([0.8]*12 + [0.2]*12 + [0.8]*12 + [0.2]*12 + [0.8]*12)
    else:
        reward_vec = np.array([0.2]*12 + [0.8]*12 + [0.2]*12 + [0.8]*12 + [0.2]*12)

            
    for subj_idx in range(nsubjects):
        beta, lr_pos, lr_neg = params[subj_idx]
        beta = norm2beta(beta)
        lr_pos = norm2alpha(lr_pos)
        lr_neg = norm2alpha(lr_neg)
            
        for b in range(nblocks): # if nblocks == 1, then use reversals
            ev[subj_idx, b, 0,:] = [.5,.5]
            for t in range(ntrials):

                # calculate choice probability
                ch_prob[subj_idx, b, t,:] = softmax(ev[subj_idx, b, t, :], beta)

                # make choice
                choices[subj_idx, b, t]   = np.random.choice([0, 1], 
                                                size=1, 
                                                p=ch_prob[subj_idx, b, t,:])[0]

                # get choice index
                if choices[subj_idx, b, t] == 0:
                    c = 0
                    choices_A[subj_idx, b, t] = 1
                    # get outcome
                    rewards[subj_idx, b, t]   = np.random.binomial(1, reward_vec[t])
                else:
                    c = 1
                    choices_A[subj_idx, b, t] = 0
                    # get outcome
                    rewards[subj_idx, b, t]   = np.random.binomial(1, 1-reward_vec[t])
                
                choice_nll[subj_idx, b, t] = np.log(ch_prob[subj_idx, b, t, c])

                # update values
                ev[subj_idx, b, t+1, :] = ev[subj_idx, b, t, :].copy()
                pe[subj_idx, b, t] = rewards[subj_idx, b, t] - ev[subj_idx, b, t, c]
                if pe[subj_idx, b, t] >= 0:
                    ev[subj_idx, b, t+1, c] += lr_pos * pe[subj_idx, b, t]
                else:
                    ev[subj_idx, b, t+1, c] += lr_neg * pe[subj_idx, b, t]

    # store params
    subj_dict = {'params'    : params,
                 'ev'        : ev, 
                 'ch_prob'   : ch_prob, 
                 'choices'   : choices, 
                 'choices_A' : choices_A, 
                 'rewards'   : rewards, 
                 'pe'        : pe, 
                 'choice_nll': choice_nll}

    return subj_dict
