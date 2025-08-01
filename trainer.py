import logging
import torch
from torch import nn
from torch.autograd.functional import jacobian

from scipy.stats.qmc import LatinHypercube

from tqdm import trange

from model import PSModel, PSModelHyper, PSModelLoRA, PSModelLoRAHyper, PSbaseModel
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from mobo.surrogate_model import GaussianProcess
from mobo.transformation import StandardTransform
from utils import *



@torch.no_grad()
def generate_ps(
    problem: any, 
    param: torch.Tensor,
    hypernet: nn.Module, 
    psmodel: nn.Module, 
    n_samples: int, 
    device: torch.device,
) -> torch.Tensor:
    """
    Generates points on the Pareto Set for one or more given task parameters.

    This function is corrected to handle both a single task parameter vector
    and a batch of task parameter vectors by iterating when necessary.

    Args:
        problem: The optimization problem instance.
        param (torch.Tensor): A tensor of task parameters. 
                              Can be a single vector of shape [n_params]
                              or a batch of vectors of shape [batch_size, n_params].
        hypernet: The trained hypernetwork.
        psmodel: The trained Pareto set model.
        n_samples (int): The number of points to generate on the Pareto set for each task parameter.
        device: The torch device to run computations on.

    Returns:
        torch.Tensor: A tensor of generated solutions.
                      - If param is a single vector, shape is [n_samples, n_dim].
                      - If param is a batch, shape is [batch_size, n_samples, n_dim].
    """
    # Set models to evaluation mode, as we are only doing inference.
    hypernet.eval()
    psmodel.eval()

    # Ensure the input parameter tensor is on the correct device.
    param = param.to(device)

    # Check if the input `param` is a single vector or a batch of vectors.
    # This makes the function flexible.
    is_batched = param.dim() > 1
    if not is_batched:
        # If it's a single vector, add a temporary batch dimension for consistent processing.
        param = param.unsqueeze(0)

    # We will loop through each parameter vector in the batch, generate its 
    # corresponding Pareto set solutions, and collect the results in a list.
    
    all_solutions = []
    for p_single in param:  # p_single has shape [n_params]
        
        # 1. Generate the weight dictionary for the single task parameter vector.
        weights_dict = hypernet(p_single)

        # 2. Sample a batch of preference vectors to cover the Pareto set.
        n_obj = problem.n_obj
        alpha = torch.ones(n_obj, device=device)
        pref_vec = torch.distributions.Dirichlet(alpha).sample((n_samples,))

        # 3. Generate the corresponding solutions using the functional psmodel.
        # The psmodel takes a batch of preferences and a single weight dictionary.
        # It returns a tensor of shape [n_samples, n_dim].
        xs_for_p = psmodel(pref_vec, weights_dict)
        
        all_solutions.append(xs_for_p)

    # Stack the list of solution tensors into a single tensor.
    # The resulting shape will be [batch_size, n_samples, n_dim].
    output = torch.stack(all_solutions)

    # If the original input was not batched, remove the batch dimension from 
    # the output to match user expectation (return shape [n_samples, n_dim]).
    if not is_batched:
        return output.squeeze(0)
    
    return output


def trainer_ppsl_bo_random(
        problem: any, 
        hpn_hidden_size: int,
        n_iter: int, 
        batch_size: int = 5,
        n_candidate: int = 1000,
        n_init: int = 30, 
        psm_hidden_size: int = 256, 
        psm_n_layer: int = 2, 
        lr_hpn: float = 1e-5,
        lr_base: float = 1e-3,
        loss_type: str = 'stch', 
        n_step: int = 500, 
        device: torch.device = 'cpu',
        lora_type: bool = True, 
        free_rank: int = 3, 
        coef_lcb: float = .01, 
        nu: float = .01,
        n_sample_params: int = 10, 
        n_sample_pref: int = 10, 
        verbose: bool = False, 
):
    n_dim, n_obj, n_params = problem.n_dim, problem.n_obj, problem.n_params
    ref_point = problem.nadir_point
    ref_point = [1.1*x for x in ref_point]
    
    # --- Initialization ---
    logging.info("Initializing dataset with Latin Hypercube Sampling.")
    # Sample in the augmented space Z = [X, T]
    sampler = LatinHypercube(d=n_dim + n_params)
    Z_init = sampler.random(n=n_init)
    
    X_init = torch.from_numpy(Z_init[:, :n_dim]).float().to(device)
    T_init = torch.from_numpy(Z_init[:, n_dim:]).float().to(device)
    
    Y_init = torch.vstack([problem.evaluate(X_init[i], T_init[i]) for i in range(n_init)])

    # Full dataset
    X_data = X_init.cpu().numpy()
    T_data = T_init.cpu().numpy()
    Y_data = Y_init.cpu().numpy()

    # Initialize models (they will be created in the first call to the trainer)
    hnet, psmodel = None, None

    for i_iter in trange(n_iter):
        logging.info(f"--- Iteration {i_iter + 1}/{n_iter} ---")

        # --- Phase 1: Update Surrogate Models ---
        logging.info("Phase 1: Updating GP surrogate models.")
        
        # Create augmented data Z = [X, T]
        Z_data = np.hstack([X_data, T_data])

        # Normalize data for GP training
        transform = StandardTransform(x_bound=[0,1])
        transform.fit(Z_data, Y_data)
        Z_norm, Y_norm = transform.do(Z_data, Y_data)

        # Train m independent GP models on augmented inputs
        # The GP model should be designed to handle n_dim + n_params inputs
        gp_model = GaussianProcess(n_var=n_dim + n_params, n_obj=n_obj, nu = 5)
        gp_model.fit(Z_norm, Y_norm)

        # --- Phase 2: Update Parametric PS Model ---
        logging.info("Phase 2: Updating Parametric PS Model.")
        # Sample n_sample_params task parameters uniformly from [pl, pu]
        params_sample = torch.rand([n_sample_params, n_params]).to(device)

        hnet, psmodel = trainer_ppsl_bo_fix_para(
            problem=problem, 
            surrogate_type='gp',
            surrogate=gp_model,
            parameters=params_sample, 
            hnet=hnet, 
            psmodel=psmodel,
            Y_norm=Y_norm,
            n_step=n_step,
            hpn_hidden_size=hpn_hidden_size, 
            psm_hidden_size=psm_hidden_size, 
            psm_n_layer=psm_n_layer, 
            lr_hpn=lr_hpn,
            lr_base=lr_base,
            device=device, 
            lora_type=lora_type, 
            loss_type=loss_type, 
            free_rank=free_rank, 
            coef_lcb=coef_lcb, 
            nu=nu,
            n_sample_pref=n_sample_pref,
            verbose=False, # Verbosity handled by the outer loop
        )

        # --- Phase 3: Acquire New Data Batch ---
        # Get current non-dominated front for HVI calculation
        nds = NonDominatedSorting()
        idx_nds = nds.do(Y_norm, only_non_dominated_front=True)
        archive_Y_norm = Y_norm[idx_nds]
        
        cand_T = torch.rand((n_candidate, n_params), device=device)
        X_new, T_new = acquire_new_data(
            problem=problem,
            gp_model=gp_model,
            hnet=hnet,
            psmodel=psmodel,
            archive_Y_norm=archive_Y_norm,
            batch_size=batch_size,
            cand_T=cand_T,
            coef_lcb=coef_lcb,
            device=device,
        )

        # Evaluate the true objectives for the new batch
        Y_new = torch.zeros((batch_size, n_obj)).to(device)
        for i in range(batch_size): 
            Y_new[i] = problem.evaluate(X_new[i], T_new[i])

        # Augment the dataset
        X_data = np.vstack([X_data, X_new.cpu().numpy()])
        T_data = np.vstack([T_data, T_new.cpu().numpy()])
        Y_data = np.vstack([Y_data, Y_new.cpu().numpy()])

        # # Optional: Calculate and log hypervolume of the true discovered points
        # test_x = torch.rand((2000, 4)).to(device)
        # test_y = torch.zeros((2000,2))
        # for i in range(2000): 
        #     test_y[i] = problem.evaluate(test_x[i, :-1], test_x[i, -1].unsqueeze(0))
        # test_y = transform.do(y=test_y.cpu().numpy())
        # test_ypre = gp_model.evaluate(test_x.cpu().numpy())['F']
        # test_ydiff = test_y - test_ypre

        # print(f"MSE for GP: 0 obj - {np.sum(test_ydiff[0] ** 2):.5f}, 1 obj - {np.sum(test_ydiff[1] ** 2):.5f}.")
        # print(f"Newly added data: {X_data}")

    return hnet, psmodel, gp_model


def trainer_ppsl_bo_fix_para(
        problem: any, 
        surrogate: any, 
        parameters: torch.Tensor, 
        hnet: None, 
        psmodel: None, 
        hpn_hidden_size: int,
        psm_hidden_size: int, 
        psm_n_layer: int, 
        Y_norm: torch.Tensor, 
        n_step: int = 100, 
        lr_hpn: float = 1e-5,
        lr_base: float = 1e-3,
        loss_type: str = 'stch', 
        device: torch.device = 'cpu',
        surrogate_type: str = 'gp', 
        coef_lcb: float = .01, 
        nu: float = .01,
        lora_type: bool = True, 
        free_rank: int = 3, 
        n_sample_pref: int = 30, 
        verbose: bool = False,
        # For hypernetwork surrogate
        surrogate_base_model: any = None,  # Base model for hypernetwork surrogate
):
    # initialize the problems' characteristics
    n_params = problem.n_params
    n_dim = problem.n_dim
    n_obj = problem.n_obj
    ideal_point = torch.tensor(problem.ideal_point).to(device)
    nadir_point = torch.tensor(problem.nadir_point).to(device)
    z = torch.min(torch.cat((torch.zeros((1,n_obj),device=device),torch.from_numpy(Y_norm).to(device) - 1.0)), axis = 0).values.data

    if hnet is None and psmodel is None: 
        if lora_type: 
            hnet = PSModelLoRAHyper(
                n_params=n_params, 
                n_dim=n_dim, 
                n_obj=n_obj, 
                free_rank=free_rank,
                params_hidden_size=hpn_hidden_size,
                psm_hidden_size=psm_hidden_size, 
                psm_n_layer=psm_n_layer,
            )
            psmodel = PSModelLoRA(
                n_dim=n_dim, 
                n_obj=n_obj, 
                free_rank=free_rank, 
                hidden_size=psm_hidden_size, 
                n_layer=psm_n_layer,
            )
        else: 
            hnet = PSModelHyper(
                n_params=n_params, 
                n_dim=n_dim, 
                n_obj=n_obj, 
                params_hidden_size=hpn_hidden_size,
                psm_hidden_size=psm_hidden_size, 
                psm_n_layer=psm_n_layer
            )
            psmodel = PSModel(
                n_dim=n_dim, 
                n_obj=n_obj, 
                hidden_size=psm_hidden_size, 
                n_layer=psm_n_layer
            )

        if verbose: 
            logging.info(f"HyperNetwork size: {count_parameters(hnet)}.")
            if lora_type: logging.info(f"Use LoRA for PS model with r: {psmodel.free_rank}.")

    hnet = hnet.to(device)
    psmodel = psmodel.to(device)

    optimizer_hpn = torch.optim.Adam(hnet.parameters(), lr=lr_hpn)
    if lora_type: optimizer_base = torch.optim.Adam(psmodel.base_model.parameters(), lr=lr_base)

    hnet.train()
    if lora_type: psmodel.base_model.train()

    params_sample = parameters.reshape(-1, n_params).to(device)

    # for _ in range(step_size): 
    total_loss = 0 

    # Iterate over the batch of task parameters
    for param in params_sample:
        for _ in range(n_step): 
            # The hypernetwork generates the task-specific weights (or LoRA adapters)
            weights = hnet(param) 

            # Sample preferences to cover the Pareto set for this task
            alpha = np.ones(n_obj)
            pref = np.random.dirichlet(alpha,n_sample_pref)
            pref_vec  = torch.tensor(pref).to(device).float() + 0.0001

            # Get the predicted solutions from the PS model
            x = psmodel(pref_vec, weights)  # [n_sample_pref, n_dim]
            x_np = x.detach().cpu().numpy()

            # Create augmented input for surrogate evaluation: [x, t]
            param_np = param.detach().cpu().numpy()
            z_np = np.concatenate([x_np, np.broadcast_to(param_np, (n_sample_pref, n_params))], axis=1)
            z_torch = torch.from_numpy(z_np).float().to(device).requires_grad_(True)

            if surrogate_type.lower() == 'gp':
                # Obtain the posterior mean and std from the GP models
                mean = torch.from_numpy(surrogate.evaluate(z_np)['F']).to(device)
                mean_grad = torch.from_numpy(surrogate.evaluate(z_np, calc_gradient=True)['dF']).to(device)

                std = torch.from_numpy(surrogate.evaluate(z_np, std=True)['S']).to(device)
                std_grad = torch.from_numpy(surrogate.evaluate(z_np, std=True, calc_gradient=True)['dS']).to(device)

                # The gradient from GP is w.r.t augmented input z, we only need the part w.r.t x
                mean_grad_x = mean_grad[:, :, :n_dim]
                std_grad_x = std_grad[:, :, :n_dim]

            elif surrogate_type.lower() == 'gp_time':
                # Obtain the posterior mean and std from the GP models
                mean = torch.from_numpy(surrogate.evaluate(x_np)['F']).to(device)
                mean_grad = torch.from_numpy(surrogate.evaluate(x_np, calc_gradient=True)['dF']).to(device)
                
                std = torch.from_numpy(surrogate.evaluate(x_np, std=True)['S']).to(device)
                std_grad = torch.from_numpy(surrogate.evaluate(x_np, std=True, calc_gradient=True)['dS']).to(device)

                # The gradient from GP is w.r.t augmented input z, we only need the part w.r.t x
                mean_grad_x = mean_grad[:, :, :n_dim]
                std_grad_x = std_grad[:, :, :n_dim]

            elif surrogate_type.lower() == 'nn':  # Neural network surrogate
                # mean prediction
                mean = surrogate(z_torch)  # [n_sample_pref, n_obj]

                # Compute gradient of each obj wrt each input variable (decision variables only)
                grads_mean = []
                for obj_idx in range(mean.shape[1]):
                    grad_outputs = torch.zeros_like(mean)
                    grad_outputs[:, obj_idx] = 1.0
                    grad = torch.autograd.grad(
                        outputs=mean, inputs=z_torch,
                        grad_outputs=grad_outputs,
                        retain_graph=True, create_graph=True, only_inputs=True
                    )[0]  # [n_sample_pref, n_dim + n_params]
                    grads_mean.append(grad[:, :n_dim])  # Only keep grads w.r.t. decision variables
                mean_grad_x = torch.stack(grads_mean, dim=1)  # [n_sample_pref, n_obj, n_dim]

                # If you do NOT use uncertainty with the neural net:
                std = torch.zeros_like(mean)
                std_grad_x = torch.zeros_like(mean_grad_x)
                
            elif surrogate_type.lower() == 'hypernet':
                
                # # Hypernetwork surrogate code
                # # surrogate is the hypernetwork, surrogate_base_model is the base model
                # x_torch = x.clone().requires_grad_(True)
                
                # # Get time-specific weights from hypernetwork
                # with torch.no_grad():
                #     surrogate_weights = surrogate(param)  # param contains time information
                
                # # Forward pass through base model with time-specific weights
                # mean = surrogate_base_model(x_torch.float(), surrogate_weights)  # [n_sample_pref, n_obj]
                
                # # Compute gradients directly w.r.t. x (no need to handle augmented input)
                # grads_mean = []
                # for obj_idx in range(mean.shape[1]):
                #     grad_outputs = torch.zeros_like(mean)
                #     grad_outputs[:, obj_idx] = 1.0
                #     grad = torch.autograd.grad(
                #         outputs=mean, inputs=x_torch,
                #         grad_outputs=grad_outputs,
                #         retain_graph=True, create_graph=True, only_inputs=True
                #     )[0]  # [n_sample_pref, n_dim]
                #     grads_mean.append(grad)
                # mean_grad_x = torch.stack(grads_mean, dim=1)  # [n_sample_pref, n_obj, n_dim]
                
                # # No uncertainty estimation for hypernetwork surrogate
                # std = torch.zeros_like(mean)
                # std_grad_x = torch.zeros_like(mean_grad_x)

                ##########UPDATE
                # Hypernetwork surrogate code
                # surrogate is the hypernetwork, surrogate_base_model is the base model
                x_torch = x.clone().requires_grad_(True)

                # Get time-specific weights from hypernetwork
                with torch.no_grad():
                    surrogate_weights = surrogate(param)  # param contains time information

                # Forward pass through base model with time-specific weights
                mean = surrogate_base_model(x_torch.float(), surrogate_weights)  # [n_sample_pref, n_obj]

                # Compute Jacobian matrix using torch.autograd.functional.jacobian
                def compute_mean(x):
                    # with torch.no_grad():  # Ensure surrogate_weights are not part of gradient computation
                    return surrogate_base_model(x.float(), surrogate_weights)

                # Compute Jacobian: d(mean)/d(x_torch)
                J = torch.autograd.functional.jacobian(
                    compute_mean,
                    x_torch,
                    create_graph=True  # To allow higher-order derivatives if needed
                )  # Shape: [n_sample_pref, n_obj, n_sample_pref, n_dim]

                # The Jacobian is block-diagonal because each output depends only on its corresponding input
                # So we extract the diagonal blocks: J[i,:,i,:] for each sample i
                mean_grad_x = torch.stack([J[i,:,i,:] for i in range(x_torch.shape[0])])  # [n_sample_pref, n_obj, n_dim]

                # No uncertainty estimation for hypernetwork surrogate
                std = torch.zeros_like(mean)
                std_grad_x = torch.zeros_like(mean_grad_x)

            else:
                raise ValueError(f"Unknown surrogate type: {surrogate_type}")

            match loss_type.lower(): 
                case 'stch': 
                    # Compute the surrogate **STCH** loss and its gradient w.r.t. x
                    loss, l_stch_grad_x = compute_stch_loss_and_gradient_stable(
                        lambda_vec=pref_vec, z_star=z, nu=nu, 
                        mean=mean, std=std, mean_grad=mean_grad_x, std_grad=std_grad_x, 
                        coef_lcb=coef_lcb,
                    ) 
                    total_loss += torch.mean(loss).item()

                case 'tch':  
                    # Compute the surrogate **TCH** loss and its gradient w.r.t. x
                    f_hat = mean - coef_lcb * std  # shape: (n_pref, m)
                    f_hat_grad = mean_grad_x - coef_lcb * std_grad_x  # shape: (n_pref, m, n_dim)
                    tch_idx = torch.argmax((1 / pref_vec) * (f_hat - z), axis = 1)
                    tch_idx_mat = [torch.arange(len(tch_idx)),tch_idx]
                    l_stch_grad_x = (1 / pref_vec)[tch_idx_mat].view(n_sample_pref,1) *  f_hat_grad[tch_idx_mat] + 0.01 * torch.sum(f_hat_grad, axis = 1) 

                    l_stch_grad_x = l_stch_grad_x / torch.norm(l_stch_grad_x, dim = 1)[:, None]

            # Backpropagate the gradient to update the PS model (and base model if LoRA)
            optimizer_hpn.zero_grad() 
            if lora_type: optimizer_base.zero_grad()

            psmodel(pref_vec, weights).backward(l_stch_grad_x) 

            if lora_type: optimizer_base.step()
            optimizer_hpn.step()
                    
    if verbose: 
        logging.info(f"STCH surrogate loss: {total_loss / len(params_sample):.4f}.")

    return hnet, psmodel

    
def acquire_new_data(
    problem: any,
    gp_model: any,
    hnet: nn.Module,
    psmodel: nn.Module,
    archive_Y_norm: np.ndarray,
    batch_size: int,
    cand_T: torch.Tensor,
    coef_lcb: float = 0.01,
    device: torch.device = 'cpu',
):
    """
    Acquires a new batch of data points for evaluation (Phase 3 of Algorithm 1).
    
    Args:
        problem: The optimization problem instance.
        gp_model: The trained Gaussian Process surrogate model.
        hnet: The trained hypernetwork.
        psmodel: The trained Pareto set model.
        archive_Y_norm: Normalized objective values of the current non-dominated solutions.
        batch_size: The number of new points to select (B).
        n_candidate: The size of the candidate pool to generate (P).
        coef_lcb: The LCB coefficient for balancing exploration/exploitation.
        device: The torch device.

    Returns:
        A tuple (X_new, T_new) of selected solutions and their corresponding task parameters.
    """
    logging.info("Phase 3: Acquiring new data batch via HVI.")
    n_dim = problem.n_dim
    n_obj = problem.n_obj
    n_params = problem.n_params
    pl, pu = problem.pl.to(device), problem.pu.to(device)
    xl, xu = problem.xl.to(device), problem.xu.to(device)

    hnet.eval()
    psmodel.eval()

    # 1. Generate Candidate Pool from the Parametric Manifold
    # Sample P >> B task-preference pairs
    # cand_T = torch.rand(n_candidate_t, n_params).to(device)
    
    alpha = torch.ones(n_obj, device=device)
    cand_pref = torch.distributions.Dirichlet(alpha).sample((1,))

    # Generate candidate solutions x_p = h(t_p, lambda_p)
    cand_X_list = []
    with torch.no_grad():
        # Get the i-th task parameter
        for t_p in cand_T:

            # Generate weights for this specific task
            weights_dict = hnet(t_p)
            
            # Generate the corresponding solution
            x_p = psmodel(cand_pref, weights_dict)
            
            cand_X_list.append(x_p)
            
        # Consolidate the list of solutions into a single tensor
        cand_X = torch.cat(cand_X_list, dim=0)

    # Clip solutions to be within bounds [xl, xu]
    cand_X = torch.max(torch.min(cand_X, torch.ones((cand_X.shape[1]),device=device)), torch.zeros((cand_X.shape[1]),device=device))

    cand_X_np = cand_X.cpu().numpy()
    cand_T_np = cand_T.cpu().numpy()
    
    # Create augmented input Z = [X, T] for the candidate pool
    cand_Z = np.hstack([cand_X_np, cand_T_np])

    # 2. Calculate LCB acquisition values for the candidate pool
    # The LCB values will be used as the objectives for HVI calculation
    out = gp_model.evaluate(cand_Z, std=True)
    mean, std = out['F'], out['S']
    cand_Y_acq = mean - coef_lcb * std

    # 3. Batched Selection via Hypervolume Improvement (HVI)
    # The reference point for HV calculation should dominate all points
    ref_point = np.max(np.vstack([archive_Y_norm, cand_Y_acq]), axis=0) + 1e-6
    hv_calculator = HV(ref_point=ref_point)

    selected_indices = []
    candidate_indices = list(range(cand_X_np.shape[0]))
    
    # Y_p is the set of points currently forming the Pareto front for HV calculation
    # It starts with the non-dominated archive and grows with each selected point.
    Y_p = archive_Y_norm
    
    for _ in range(batch_size):
        max_hv_improvement = -1
        best_idx_in_cand = -1

        # Calculate current hypervolume
        hv_current = hv_calculator.do(Y_p)

        # Find the point that maximizes marginal HVI
        for idx in candidate_indices:
            # Combine Y_p with the acquisition value of the candidate point
            Y_combined = np.vstack([Y_p, cand_Y_acq[idx]])
            hv_after = hv_calculator.do(Y_combined)
            hv_improvement = hv_after - hv_current

            if hv_improvement > max_hv_improvement:
                max_hv_improvement = hv_improvement
                best_idx_in_cand = idx
        
        if best_idx_in_cand == -1:
            # This can happen if all candidates are dominated or identical
            # Fallback: select a random remaining candidate
            if not candidate_indices: break
            best_idx_in_cand = np.random.choice(candidate_indices)

        # Add the best point to the batch and update Y_p for the next selection
        selected_indices.append(best_idx_in_cand)
        candidate_indices.remove(best_idx_in_cand)
        Y_p = np.vstack([Y_p, cand_Y_acq[best_idx_in_cand]])

    X_new = torch.from_numpy(cand_X_np[selected_indices]).to(device)
    T_new = torch.from_numpy(cand_T_np[selected_indices]).to(device)

    logging.info(f"Selected {len(selected_indices)} new points for evaluation.")
    return X_new, T_new


def trainer_psmodel(
        problem: any, 
        psm_hidden_size: int, 
        psm_n_layer: int, 
        n_epochs: int, 
        lr: float,
        loss_type: str, 
        device: torch.device,
        n_sample_pref: int = 30, 
):
    n_dim = problem.n_dim
    n_obj = problem.n_obj
    ideal_point = torch.tensor(problem.ideal_point).to(device)
    nadir_point = torch.tensor(problem.nadir_point).to(device)
    z = torch.zeros(n_obj).to(device)
    
    psmodel = PSbaseModel(
        n_dim=n_dim, 
        n_obj=n_obj, 
        hidden_size=psm_hidden_size, 
        n_layer=psm_n_layer
    )

    psmodel = psmodel.to(device)
    optimizer = torch.optim.Adam(psmodel.parameters(), lr=lr)

    epoch_iter = trange(n_epochs)
    for epoch in epoch_iter: 
        psmodel.train()
            
        # sample n_pref_update preferences
        alpha = torch.ones(n_obj, device=device)
        pref = torch.distributions.Dirichlet(alpha).sample((n_sample_pref,))
        pref_vec = pref
        
        # get the predicted Pareto set 
        x = psmodel(pref_vec) 
        
        # evaluate the loss 
        value = problem.evaluate(x)
        value = (value - ideal_point) / (nadir_point - ideal_point)

        # scalarize the objectives
        loss = loss_func(type=loss_type, 
                        preference_vector=pref_vec, 
                        func_value=value, 
                        z=z)

        # gradient descent
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step()

    return psmodel