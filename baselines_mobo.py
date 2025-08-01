"""
Runing the proposed Paret Set Learning (PSL) method on 15 test problems.
"""

import numpy as np
import torch
import torch.nn as nn
import pickle

from tqdm import trange

from scipy.stats.qmc import LatinHypercube

from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from mobo.surrogate_model import GaussianProcess
from mobo.transformation import StandardTransform

from utils import compute_stch_loss_and_gradient_stable


class ParetoSetModel(torch.nn.Module):
    def __init__(self, n_dim, n_obj):
        super(ParetoSetModel, self).__init__()
        self.n_dim = n_dim
        self.n_obj = n_obj
       
        self.fc1 = nn.Linear(self.n_obj, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, self.n_dim)
       
    def forward(self, pref):

        x = torch.relu(self.fc1(pref))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        x = torch.sigmoid(x) 
        
        return x.to(torch.float64)
    

def psl_mobo(
        problem, 
        n_iter: int, 
        n_sample: int, 
        n_init: int = 20, 
        n_steps: int = 1000, 
        n_pref_update = 10,
        coef_lcb = 0.1, 
        n_candidate = 1000, 
        n_local = 1, 
        device = 'cuda', 
        verbose: bool = False,
):
# -----------------------------------------------------------------------------
    # get problem info
    hv_all_value = np.zeros([n_iter])
    n_dim = problem.n_dim
    n_obj = problem.n_obj

    ref_point = problem.nadir_point
    ref_point = [1.1*x for x in ref_point]
                    
    # initialize n_init solutions 
    sampler = LatinHypercube(d=n_dim)
    x_init = sampler.random(n=n_init)
    y_init = problem.evaluate(torch.from_numpy(x_init).to(device))

    X = x_init
    Y = y_init.cpu().numpy()

    z = torch.zeros(n_obj).to(device)

    iterable = trange(n_iter) if verbose else range(n_iter)
    # n_iter batch selections 
    for i_iter in iterable:
        
        # intitialize the model and optimizer 
        psmodel = ParetoSetModel(n_dim, n_obj)
        psmodel.to(device)
            
        # optimizer
        optimizer = torch.optim.Adam(psmodel.parameters(), lr=1e-3)
        
        # solution normalization
        transformation = StandardTransform([0,1])
        transformation.fit(X, Y)
        X_norm, Y_norm = transformation.do(X, Y) 
        
        # train GP surrogate model 
        surrogate_model = GaussianProcess(n_dim, n_obj, nu = 5)
        surrogate_model.fit(X_norm,Y_norm)
        
        z =  torch.min(torch.cat((z.reshape(1,n_obj),torch.from_numpy(Y_norm).to(device) - 0.1)), axis = 0).values.data
        
        # nondominated X, Y 
        nds = NonDominatedSorting()
        idx_nds = nds.do(Y_norm)
        
        X_nds = X_norm[idx_nds[0]]
        Y_nds = Y_norm[idx_nds[0]]
        
        # t_step Pareto Set Learning with Gaussian Process
        for t_step in range(n_steps):
            psmodel.train()
            
            # sample n_pref_update preferences
            alpha = np.ones(n_obj)
            pref = np.random.dirichlet(alpha,n_pref_update)
            pref_vec  = torch.tensor(pref).to(device).float() + 0.0001
            
            # get the current coressponding solutions
            x = psmodel(pref_vec)
            x_np = x.detach().cpu().numpy()
            
            # obtain the value/grad of mean/std for each obj
            mean = torch.from_numpy(surrogate_model.evaluate(x_np)['F']).to(device)
            mean_grad = torch.from_numpy(surrogate_model.evaluate(x_np, calc_gradient=True)['dF']).to(device)
            
            std = torch.from_numpy(surrogate_model.evaluate(x_np, std=True)['S']).to(device)
            std_grad = torch.from_numpy(surrogate_model.evaluate(x_np, std=True, calc_gradient=True)['dS']).to(device)
            
            # calculate the value/grad of tch decomposition with LCB
            # value = mean - coef_lcb * std
            # value_grad = mean_grad - coef_lcb * std_grad
            
            # tch_idx = torch.argmax((1 / pref_vec) * (value - z), axis = 1)
            # tch_idx_mat = [torch.arange(len(tch_idx)),tch_idx]
            # tch_grad = (1 / pref_vec)[tch_idx_mat].view(n_pref_update,1) *  value_grad[tch_idx_mat] + 0.01 * torch.sum(value_grad, axis = 1) 

            # tch_grad = tch_grad / torch.norm(tch_grad, dim = 1)[:, None]
            loss, tch_grad = compute_stch_loss_and_gradient_stable(
                lambda_vec=pref_vec, z_star=z, nu=.01, mean=mean, std=std, 
                mean_grad=mean_grad, std_grad=std_grad, coef_lcb=.1, 
            )
            
            # gradient-based pareto set model update 
            optimizer.zero_grad()
            psmodel(pref_vec).backward(tch_grad)
            optimizer.step()  
            
        # solutions selection on the learned Pareto set
        psmodel.eval()
        
        # sample n_candidate preferences
        alpha = np.ones(n_obj)
        pref = np.random.dirichlet(alpha,n_candidate)
        pref  = torch.tensor(pref).to(device).float() + 0.0001

        # generate correponding solutions, get the predicted mean/std
        X_candidate = psmodel(pref).to(torch.float64)
        X_candidate_np = X_candidate.detach().cpu().numpy()
        Y_candidate_mean = surrogate_model.evaluate(X_candidate_np)['F']
        
        Y_candidata_std = surrogate_model.evaluate(X_candidate_np, std=True)['S']
        Y_candidate = Y_candidate_mean - coef_lcb * Y_candidata_std
        
        # optional TCH-based local Exploitation 
        if n_local > 0:
            X_candidate_tch = X_candidate_np
            z_candidate = z.cpu().numpy()
            pref_np = pref.cpu().numpy()
            for j in range(n_local):
                candidate_mean =  surrogate_model.evaluate(X_candidate_tch)['F']
                candidate_mean_grad =  surrogate_model.evaluate(X_candidate_tch, calc_gradient=True)['dF']
                
                candidate_std = surrogate_model.evaluate(X_candidate_tch, std=True)['S']
                candidate_std_grad = surrogate_model.evaluate(X_candidate_tch, std=True, calc_gradient=True)['dS']
                
                candidate_value = candidate_mean - coef_lcb * candidate_std
                candidate_grad = candidate_mean_grad - coef_lcb * candidate_std_grad
                
                candidate_tch_idx = np.argmax((1 / pref_np) * (candidate_value - z_candidate), axis = 1)
                candidate_tch_idx_mat = [np.arange(len(candidate_tch_idx)),list(candidate_tch_idx)]
                
                candidate_tch_grad = (1 / pref_np)[np.arange(len(candidate_tch_idx)),list(candidate_tch_idx)].reshape(n_candidate,1) * candidate_grad[np.arange(len(candidate_tch_idx)),list(candidate_tch_idx)] 
                candidate_tch_grad +=  0.01 * np.sum(candidate_grad, axis = 1) 
                
                X_candidate_tch = X_candidate_tch - 0.01 * candidate_tch_grad
                X_candidate_tch[X_candidate_tch <= 0]  = 0
                X_candidate_tch[X_candidate_tch >= 1]  = 1  
                
            X_candidate_np = np.vstack([X_candidate_np, X_candidate_tch])
            
            Y_candidate_mean = surrogate_model.evaluate(X_candidate_np)['F']
            Y_candidata_std = surrogate_model.evaluate(X_candidate_np, std=True)['S']
            
            Y_candidate = Y_candidate_mean - coef_lcb * Y_candidata_std
        
        # greedy batch selection 
        best_subset_list = []
        Y_p = Y_nds 
        for b in range(n_sample):
            hv = HV(ref_point=np.max(np.vstack([Y_p,Y_candidate]), axis = 0))
            best_hv_value = 0
            best_subset = None
            
            for k in range(len(Y_candidate)):
                Y_subset = Y_candidate[k]
                Y_comb = np.vstack([Y_p,Y_subset])
                hv_value_subset = hv(Y_comb)
                if hv_value_subset > best_hv_value:
                    best_hv_value = hv_value_subset
                    best_subset = [k]
            
            if best_subset == None: best_subset = [np.random.randint(0, len(Y_candidate))]
            Y_p = np.vstack([Y_p,Y_candidate[best_subset]])
            best_subset_list.append(best_subset)  
            
        best_subset_list = np.array(best_subset_list).T[0]
        
        # evaluate the selected n_sample solutions
        X_candidate = torch.tensor(X_candidate_np).to(device)
        X_new = X_candidate[best_subset_list]
        Y_new = problem.evaluate(X_new)
        
        # update the set of evaluated solutions (X,Y)
        X = np.vstack([X,X_new.detach().cpu().numpy()])
        Y = np.vstack([Y,Y_new.detach().cpu().numpy()])
        
        # check the current HV for evaluated solutions
        hv = HV(ref_point=np.array(ref_point))
        hv_all_value[i_iter] = hv(Y)
        
    return psmodel, hv_all_value


import torch
import numpy as np
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
)
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.acquisition.objective import GenericMCObjective
from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.utils.sampling import sample_simplex

def mobo_botorch(
    problem,
    n_iter: int,
    n_sample: int,
    n_init: int = 20,
    algo: str = "qehvi",  # "qehvi" or "qnparego"
    noise_se: list = None,  # If known, pass as list of std for each obj, otherwise None
    dtype=torch.double,
    device=None,
    seed: int = 0,
):
    """
    Multi-objective Bayesian Optimization using BoTorch with qEHVI or qNParEGO.

    Parameters
    ----------
    problem : object
        Must have .bounds (2, n_var), .evaluate(X) (N, n_var)->(N, n_obj), .n_var, .n_obj, .ref_point (list/array)
    n_iter : int
        Number of BO iterations (batches)
    n_sample : int
        Batch size (number of new points per iteration)
    n_init : int
        Number of initial samples (Sobol)
    algo : str
        "qehvi" or "qnparego"
    noise_se : list or None
        Observation noise std for each objective; if None, GP will infer noise.
    dtype : torch.dtype
        Data type for tensors.
    device : torch.device or None
        Device to use; if None, use cuda if available.
    seed : int
        Random seed.

    Returns
    -------
    hvs : np.ndarray
        Hypervolume history (n_iter+1,)
    Y_hist : np.ndarray
        All observed true objectives, shape (n_init + n_iter*n_sample, n_obj)
    """

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tkwargs = {"dtype": dtype, "device": device}

    n_obj, n_var = problem.n_obj, problem.n_var

    # 1. Generate initial data
    def generate_initial_data(n):
        train_x = draw_sobol_samples(
            bounds=torch.tensor(problem.bounds, **tkwargs),
            n=n,
            q=1,
        ).squeeze(1)
        with torch.no_grad():
            train_obj_true = torch.tensor(
                problem.evaluate(train_x.cpu().numpy()), **tkwargs
            )
        if noise_se is not None:
            train_obj = train_obj_true + torch.randn_like(train_obj_true) * torch.tensor(noise_se, **tkwargs)
        else:
            train_obj = train_obj_true
        return train_x, train_obj, train_obj_true

    # 2. GP model initialization
    def initialize_model(train_x, train_obj):
        train_x_norm = normalize(train_x, torch.tensor(problem.bounds, **tkwargs))
        models = []
        for i in range(n_obj):
            train_y = train_obj[..., i : i + 1]
            if noise_se is not None:
                train_yvar = torch.full_like(train_y, noise_se[i] ** 2)
            else:
                train_yvar = None
            models.append(
                SingleTaskGP(
                    train_x_norm, train_y, train_yvar, outcome_transform=Standardize(m=1)
                )
            )
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        return mll, model

    # 3. Utility: get HV
    def get_hv(Y):
        # Only non-dominated points
        Y = Y.cpu().numpy() if isinstance(Y, torch.Tensor) else Y
        mask = is_non_dominated(torch.tensor(Y, **tkwargs)).cpu().numpy()
        Y_nd = Y[mask]
        bd = DominatedPartitioning(
            ref_point=torch.tensor(problem.ref_point, **tkwargs), Y=torch.tensor(Y_nd, **tkwargs)
        )
        return bd.compute_hypervolume().item()

    # 4. qEHVI step
    def step_qehvi(model, train_x, train_obj, sampler):
        with torch.no_grad():
            pred = model.posterior(normalize(train_x, torch.tensor(problem.bounds, **tkwargs))).mean
        partitioning = FastNondominatedPartitioning(
            ref_point=torch.tensor(problem.ref_point, **tkwargs), Y=pred
        )
        acq_func = qExpectedHypervolumeImprovement(
            model=model,
            ref_point=problem.ref_point,
            partitioning=partitioning,
            sampler=sampler,
        )
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=torch.zeros(2, n_var, **tkwargs).fill_(0).index_add_(
                0, torch.tensor([1], device=device), torch.ones(1, n_var, **tkwargs)
            ),
            q=n_sample,
            num_restarts=5,
            raw_samples=256,
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True,
        )
        new_x = unnormalize(candidates.detach(), bounds=torch.tensor(problem.bounds, **tkwargs))
        with torch.no_grad():
            new_obj_true = torch.tensor(
                problem.evaluate(new_x.cpu().numpy()), **tkwargs
            )
        if noise_se is not None:
            new_obj = new_obj_true + torch.randn_like(new_obj_true) * torch.tensor(noise_se, **tkwargs)
        else:
            new_obj = new_obj_true
        return new_x, new_obj, new_obj_true

    # 5. qNParEGO step (parallel Chebyshev scalarizations)
    def step_qnparego(model, train_x, train_obj, sampler):
        train_x_norm = normalize(train_x, torch.tensor(problem.bounds, **tkwargs))
        with torch.no_grad():
            pred = model.posterior(train_x_norm).mean
        acq_func_list = []
        for _ in range(n_sample):
            weights = sample_simplex(n_obj, **tkwargs).squeeze()
            objective = GenericMCObjective(
                get_chebyshev_scalarization(weights=weights, Y=pred)
            )
            acq_func = qNoisyExpectedImprovement(
                model=model,
                objective=objective,
                X_baseline=train_x_norm,
                sampler=sampler,
                prune_baseline=True,
            )
            acq_func_list.append(acq_func)
        candidates, _ = optimize_acqf_list(
            acq_function_list=acq_func_list,
            bounds=torch.zeros(2, n_var, **tkwargs).fill_(0).index_add_(
                0, torch.tensor([1], device=device), torch.ones(1, n_var, **tkwargs)
            ),
            num_restarts=5,
            raw_samples=256,
            options={"batch_limit": 5, "maxiter": 200},
        )
        new_x = unnormalize(candidates.detach(), bounds=torch.tensor(problem.bounds, **tkwargs))
        with torch.no_grad():
            new_obj_true = torch.tensor(
                problem.evaluate(new_x.cpu().numpy()), **tkwargs
            )
        if noise_se is not None:
            new_obj = new_obj_true + torch.randn_like(new_obj_true) * torch.tensor(noise_se, **tkwargs)
        else:
            new_obj = new_obj_true
        return new_x, new_obj, new_obj_true

    # --- Main loop ---
    train_x, train_obj, train_obj_true = generate_initial_data(n_init)
    mll, model = initialize_model(train_x, train_obj)
    hvs = [get_hv(train_obj_true)]

    for i in range(n_iter):
        # Fit GP
        from botorch import fit_gpytorch_mll
        fit_gpytorch_mll(mll)

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))

        if algo.lower() == "qehvi":
            new_x, new_obj, new_obj_true = step_qehvi(model, train_x, train_obj, sampler)
        elif algo.lower() == "qnparego":
            new_x, new_obj, new_obj_true = step_qnparego(model, train_x, train_obj, sampler)
        else:
            raise ValueError(f"Unknown algo: {algo}, should be 'qehvi' or 'qnparego'.")

        # Update data
        train_x = torch.cat([train_x, new_x])
        train_obj = torch.cat([train_obj, new_obj])
        train_obj_true = torch.cat([train_obj_true, new_obj_true])

        # HV
        hvs.append(get_hv(train_obj_true))

        # Re-init model (no warm start)
        mll, model = initialize_model(train_x, train_obj)

    return np.array(hvs), train_obj_true