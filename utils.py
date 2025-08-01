import torch 
from torch import Tensor

import numpy as np 
import random  


def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def loss_func(type: str, preference_vector: Tensor, func_value: Tensor, z: Tensor, mu: float = 0.01): 
    agg_value = agge_func(type, preference_vector, func_value, z, mu)
    return  torch.sum(agg_value)


def agge_func(type: str, preference_vector: Tensor, func_value: Tensor, z: Tensor, mu: float = 0.01): 
    
    match type.lower(): 
        case 'ls': 
            agg_value = torch.sum(preference_vector * (func_value - z), dim=1)
        
        case 'tch': 
            agg_value =  torch.max(preference_vector * (func_value - z), dim=1)[0] 
        
        case 'stch': 
            agg_value = mu* torch.logsumexp(preference_vector * (func_value - z) / mu, dim=1)   
            
    return agg_value
        

@torch.no_grad()
def gradient_estimation(problem, x, param, loss_type, preference_vecotr, z, n_grad_esti, sigma: float = 0.02): 
    n_x, n_dim = x.shape
    device = x.device.type
    
    # fitness evaluation
    # tmps = torch.arange(1, n_grad_esti+1)
    # denominator = torch.max(torch.tensor(0), torch.log(torch.tensor(1+n_grad_esti/2))-torch.log(tmps)).sum()
    # r = torch.max(torch.tensor(0), torch.log(torch.tensor(1+n_grad_esti/2))-torch.log(tmps)) / denominator - 1/n_grad_esti
    r = torch.linspace(0.5, -0.5, n_grad_esti)
    
    # gradient
    grad = torch.zeros(x.shape, device=device)
    for i in range(n_x):
        x_tmp, pref_tmp = x[i], preference_vecotr[i]
        # sampled u_k
        sampled_dire = torch.bernoulli(.5 * torch.ones((n_grad_esti, n_dim), device=device))
        
        # evaluate r_k
        func_values = problem.evaluate(x_tmp + sigma*sampled_dire, param) 
        agg_value = agge_func(loss_type, pref_tmp, func_values, z)
        
        # obtain r_k
        _, sorted_indices = torch.sort(agg_value, descending=True)
        r_sorted = r[sorted_indices].reshape(-1, 1)
        
        # compute the estimated gradient
        grad[i] = torch.sum(r_sorted * sampled_dire, dim=0) / (sigma * n_grad_esti)
    
    return grad


@torch.no_grad()
def compute_stch_loss_and_gradient_stable(lambda_vec, z_star, nu, 
                                         mean, std, mean_grad, std_grad, coef_lcb, 
                                         epsilon = 0):
    """
    Numerically stable version using log-sum-exp trick.
    """    
    # Compute f_hat and its gradient
    f_hat = mean - coef_lcb * std  # shape: (n_pref, m)
    f_hat_grad = mean_grad - coef_lcb * std_grad  # shape: (n_pref, m, n_dim)
    
    # Compute the exponent terms
    target_shifted = z_star - epsilon  # shape: (m,)
    diff = f_hat - target_shifted.unsqueeze(0)  # shape: (n_pref, m)
    exponent = (lambda_vec * diff) / nu  # shape: (n_pref, m)
    
    # Use log-sum-exp for numerical stability
    max_exp = torch.max(exponent, dim=1, keepdim=True)[0]  # shape: (n_pref, 1)
    exp_terms_stable = torch.exp(exponent - max_exp)  # shape: (n_pref, m)
    sum_exp_stable = torch.sum(exp_terms_stable, dim=1)  # shape: (n_pref,)
    
    # Compute the loss
    loss = nu * (torch.log(sum_exp_stable) + max_exp.squeeze(1))
    
    # Compute the gradient
    weights = exp_terms_stable / sum_exp_stable.unsqueeze(1)  # shape: (n_pref, m)
    weighted_lambda = weights * lambda_vec  # shape: (n_pref, m)
    
    # Sum over objectives using einsum
    loss_grad = torch.einsum('pm,pmn->pn', weighted_lambda.to(torch.float32), f_hat_grad.to(torch.float32))
    
    return loss, loss_grad


def sample_historical_data(historical_data, sampling_strategy, max_samples, 
                         current_gen, decay_factor, recent_window, device):
    """
    Sample from historical data based on different strategies
    """
    
    all_X, all_T, all_Y, all_gens = [], [], [], []
    
    for X, T, Y, gen in historical_data:
        all_X.append(X)
        all_T.append(T)
        all_Y.append(Y)
        all_gens.extend([gen] * X.shape[0])
    
    if not all_X:
        return None, None, None
    
    # Concatenate all historical data
    X_all = torch.cat(all_X, dim=0)
    T_all = torch.cat(all_T, dim=0)
    Y_all = torch.cat(all_Y, dim=0)
    gens_all = torch.tensor(all_gens, device=device)
    
    n_total = X_all.shape[0]
    
    if sampling_strategy == 'uniform':
        # Uniform random sampling
        if n_total > max_samples:
            indices = torch.randperm(n_total, device=device)[:max_samples]
        else:
            indices = torch.arange(n_total, device=device)
            
    elif sampling_strategy == 'recent':
        # Sample only from recent time steps
        recent_mask = gens_all >= (current_gen - recent_window)
        recent_indices = torch.where(recent_mask)[0]
        
        if len(recent_indices) > max_samples:
            perm = torch.randperm(len(recent_indices), device=device)
            indices = recent_indices[perm[:max_samples]]
        else:
            indices = recent_indices
            
    elif sampling_strategy == 'exponential':
        # Exponentially decaying probability based on age
        ages = current_gen - gens_all.float()
        weights = torch.exp(-ages * (1 - decay_factor))
        
        # Sample with replacement based on weights
        if n_total > max_samples:
            indices = torch.multinomial(weights, max_samples, replacement=True)
        else:
            indices = torch.arange(n_total, device=device)
            
    elif sampling_strategy == 'weighted':
        # Linear weighting based on recency
        ages = current_gen - gens_all.float()
        max_age = ages.max()
        weights = 1.0 - (ages / (max_age + 1))
        
        if n_total > max_samples:
            indices = torch.multinomial(weights, max_samples, replacement=True)
        else:
            indices = torch.arange(n_total, device=device)
    
    else:
        raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
    
    # Return sampled data
    return X_all[indices], T_all[indices], Y_all[indices]