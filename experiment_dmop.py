import os
import time
import pickle
from tqdm import trange

import numpy as np
import torch
import torch.nn as nn

from pymoo.algorithms.moo.dnsga2 import DNSGA2
from pymoo.core.callback import CallbackCollection, Callback
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.problems.dyn import TimeSimulation
from pymoo.termination import get_termination
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import HV

from model import fxModelLoRAHyper, fxModelLoRA
from problems.problem import mop_dyn
from trainer import trainer_ppsl_bo_fix_para, generate_ps
from mobo.surrogate_model import GaussianProcess
from mobo.transformation import StandardTransform
from utils import sample_historical_data

from datetime import datetime



def run_dmop_ppsl_bo(
    problem, max_n_gen, tau_t, pop_size=100, n_init=20, batch_size=5, n_candidate=1000,
    hpn_hidden_size=1024, psm_hidden_size=256, psm_n_layer=2,
    lr_hpn=1e-5, lr_base=1e-3,
    coef_lcb = 0.05,
    loss_type='stch', lora_type=True, free_rank=3, n_sample_pref=20,
    device='cpu', verbose=False,
    surrogate_hidden_dim=128, surrogate_n_layers=3, surrogate_epochs=100,
    # GP data sampling parameters
    max_gp_samples=100,  # Total maximum samples for GP
    current_ratio=1,   # Ratio of current vs past data (0.5 means 50/50)
    past_window=2,       # Number of past generations to consider
):
    import numpy as np
    import torch
    from tqdm import trange
    from pymoo.indicators.hv import HV
    from pymoo.indicators.igd import IGD

    n_dim, n_obj = problem.n_dim, problem.n_obj
    n_iter = int((pop_size - n_init) / batch_size) + 1

    X_data, T_data, Y_data = [], [], []
    igd_all, hv_all, igds, hvs = [], [], [], []

    hnet, psmodel, surrogate = None, None, None

    for gen in trange(max_n_gen, desc="DMOP PPSL-BO-NN"):
        # Initialize at each time step
        sampler = torch.distributions.uniform.Uniform(0, 1)
        X_init = sampler.sample((n_init, n_dim)).to(device)
        T_init = torch.ones((n_init, 1), device=device) * (gen / (max_n_gen + 1))
        Y_init = problem.evaluate(X_init)

        X_data.append(X_init)
        T_data.append(T_init)
        Y_data.append(Y_init)

        t_norm = float(gen) / (max_n_gen + 1)

        for i_iter in range(n_iter): 
            # Prepare data for surrogate with limited samples
            # Calculate how many samples to use from current and past
            max_current_samples = int(max_gp_samples * current_ratio)
            max_past_samples = max_gp_samples - max_current_samples
            
            # Get current generation data
            X_current = X_data[gen]
            T_current = T_data[gen]
            Y_current = Y_data[gen]
            
            # Sample from current generation if needed
            n_current = X_current.shape[0]
            if n_current > max_current_samples:
                current_indices = np.random.choice(n_current, max_current_samples, replace=False)
                X_current_sampled = X_current[current_indices]
                T_current_sampled = T_current[current_indices]
                Y_current_sampled = Y_current[current_indices]
            else:
                X_current_sampled = X_current
                T_current_sampled = T_current
                Y_current_sampled = Y_current
                # Adjust past samples if current has fewer samples
                max_past_samples = max_gp_samples - n_current
            
            # Collect past data if available
            if gen > 0 and max_past_samples > 0:
                X_past_list = []
                T_past_list = []
                Y_past_list = []
                
                # Determine which past generations to consider
                start_gen = max(0, gen - past_window)
                
                for past_gen in range(start_gen, gen):
                    X_past_list.append(X_data[past_gen])
                    T_past_list.append(T_data[past_gen])
                    Y_past_list.append(Y_data[past_gen])
                
                if X_past_list:
                    X_past_all = torch.cat(X_past_list, dim=0)
                    T_past_all = torch.cat(T_past_list, dim=0)
                    Y_past_all = torch.cat(Y_past_list, dim=0)
                    
                    # Sample from past data if needed
                    n_past = X_past_all.shape[0]
                    if n_past > max_past_samples:
                        past_indices = np.random.choice(n_past, max_past_samples, replace=False)
                        X_past_sampled = X_past_all[past_indices]
                        T_past_sampled = T_past_all[past_indices]
                        Y_past_sampled = Y_past_all[past_indices]
                    else:
                        X_past_sampled = X_past_all
                        T_past_sampled = T_past_all
                        Y_past_sampled = Y_past_all
                    
                    # Combine current and past data
                    X_all = torch.cat([X_current_sampled, X_past_sampled], dim=0)
                    T_all = torch.cat([T_current_sampled, T_past_sampled], dim=0)
                    Y_all = torch.cat([Y_current_sampled, Y_past_sampled], dim=0)
                else:
                    X_all = X_current_sampled
                    T_all = T_current_sampled
                    Y_all = Y_current_sampled
            else:
                X_all = X_current_sampled
                T_all = T_current_sampled
                Y_all = Y_current_sampled
            
            # Create augmented input Z = [X, T]
            Z_all = torch.cat([X_all, T_all], dim=1)
            Z_np = Z_all.cpu().numpy()
            Y_np = Y_all.cpu().numpy()
            
            # Normalize data
            transform = StandardTransform(x_bound=[0,1])
            transform.fit(Z_np, Y_np)
            Z_norm, Y_norm = transform.do(Z_np, Y_np)

            # Fit Gaussian Process with augmented input
            if surrogate is None:
                surrogate = GaussianProcess(n_var=n_dim+1, n_obj=n_obj, nu=5)  # n_dim+1 for [x,t]
            surrogate.fit(X=Z_norm, Y=Y_norm)

            # ========== RMSE Evaluation ==========
            n_test_points = 2000
            # Generate random test points at current time
            X_test = sampler.sample((n_test_points, n_dim)).to(device)
            T_test = torch.ones((n_test_points, 1), device=device) * t_norm
            Y_test_true = problem.evaluate(X_test).cpu().numpy()
            
            # Create augmented test input
            Z_test = torch.cat([X_test, T_test], dim=1).cpu().numpy()
            Z_test_norm = transform.do(Z_test)
            
            # Predict with surrogate
            Y_test_pred_norm = surrogate.evaluate(Z_test_norm)['F']
            Y_test_pred = transform.undo(y=Y_test_pred_norm)
            
            # Calculate RMSE
            mse_per_obj = np.mean((Y_test_true - Y_test_pred)**2, axis=0)
            rmse_per_obj = np.sqrt(mse_per_obj)
            if verbose:
                print(f"Problem: {problem.p.__class__.__name__}, generation: {gen} | "
                      f"iteration: {i_iter} | RMSE: {rmse_per_obj.round(5)} | "
                      f"GP samples: {Z_all.shape[0]} (current: {X_current_sampled.shape[0]}, "
                      f"past: {Z_all.shape[0] - X_current_sampled.shape[0]})")
            
            # 3. PPSL model for current time
            t_tensor = torch.full((2, 1), t_norm, device=device)  # degenerate batch
            
            # Use only current generation Y_norm for PPSL training
            Y_current_norm = Y_norm[:X_current_sampled.shape[0]]  # First part is current data
            
            hnet, psmodel = trainer_ppsl_bo_fix_para(
                problem=problem,
                surrogate=surrogate,
                surrogate_type='gp',
                parameters=t_tensor,
                hnet=hnet,
                psmodel=psmodel,
                hpn_hidden_size=hpn_hidden_size,
                psm_hidden_size=psm_hidden_size,
                psm_n_layer=psm_n_layer,
                Y_norm=Y_current_norm,  # Use only current data for reference point
                n_step=20,
                lr_hpn=lr_hpn,
                lr_base=lr_base,
                loss_type=loss_type,
                device=device,
                coef_lcb=coef_lcb,
                nu=0.1,
                lora_type=lora_type,
                free_rank=free_rank,
                n_sample_pref=n_sample_pref,
                verbose=verbose,
            )

            # Data acquisition: only points at current t for HVI
            Y_cur_t_norm = Y_current_norm

            # Candidate pool (all with current t)
            cand_T = torch.full((2*n_candidate, 1), t_norm, device=device)
            cand_ps = generate_ps(problem, cand_T[0], hnet, psmodel, n_samples=n_candidate, device=device)
            cand_X = cand_ps.squeeze(1) if cand_ps.dim() == 3 else cand_ps  # [n_candidate, n_dim]
            cand_X = torch.cat([cand_X, sampler.sample((n_candidate, n_dim)).to(device)])

            # Surrogate prediction for candidate pool
            cand_Z = torch.cat([cand_X, cand_T], dim=1).cpu().numpy()
            cand_Z_norm = transform.do(cand_Z)
            cand_mean_norm = surrogate.evaluate(cand_Z_norm)['F']
            cand_std_norm = surrogate.evaluate(cand_Z_norm, std=True)['S']
            cand_value_norm = cand_mean_norm - coef_lcb * cand_std_norm

            # HVI selection batch (using HV over current t only)
            ref_point = np.max(np.vstack([Y_cur_t_norm, cand_value_norm]), axis=0) + 1e-6
            hv_calculator = HV(ref_point=ref_point)
            selected_indices = []
            candidate_indices = list(range(cand_X.shape[0]))
            Y_p = Y_cur_t_norm

            for _ in range(batch_size):
                max_hv_improvement = -1
                best_idx = -1
                hv_current = hv_calculator.do(Y_p)
                for idx in candidate_indices:
                    Y_combined = np.vstack([Y_p, cand_value_norm[idx]])
                    hv_after = hv_calculator.do(Y_combined)
                    hv_improvement = hv_after - hv_current
                    if hv_improvement > max_hv_improvement:
                        max_hv_improvement = hv_improvement
                        best_idx = idx
                if best_idx == -1:
                    best_idx = np.random.choice(candidate_indices)
                selected_indices.append(best_idx)
                candidate_indices.remove(best_idx)
                Y_p = np.vstack([Y_p, cand_value_norm[best_idx]])

            # 6. Evaluate true objectives for batch
            X_new = cand_X[selected_indices]
            T_new = cand_T[selected_indices]
            Y_new = problem.evaluate(X_new)

            X_data[gen] = torch.vstack([X_data[gen], X_new])
            T_data[gen] = torch.vstack([T_data[gen], T_new])
            Y_data[gen] = torch.vstack([Y_data[gen], Y_new])

        # 7. Generate PPSL Pareto set (for metric computation)
        ps_solutions = generate_ps(
            problem=problem,
            param=torch.tensor([t_norm], device=device),
            hypernet=hnet,
            psmodel=psmodel,
            n_samples=1000,
            device=device
        )
        F_pred = problem.evaluate(ps_solutions).cpu().numpy()
        PF_true = problem._calc_pareto_front(n_pareto_points=pop_size)

        igd = IGD(PF_true).do(F_pred)
        hv = HV(pf=PF_true)(F_pred)
        igd_all.append(igd)
        hv_all.append(hv)

        if (gen + 1) % tau_t == 0: 
            igds.append(igd)
            hvs.append(hv)

        problem.tic()

        if verbose:
            print(f"Problem: {problem.p.__class__.__name__}, Gen {gen}: t={t_norm:.3f}, IGD={igd:.4e}, HV={hv:.4f}")

    return {
        'IGD': np.array(igds),
        'HV': np.array(hvs),
        'X_data': [x.cpu().numpy() for x in X_data],
        'T_data': [t.cpu().numpy() for t in T_data],
        'Y_data': [y.cpu().numpy() for y in Y_data],
        'IGD_ALL': np.array(igd_all), 
        'HV_ALL': np.array(hv_all)
    }


def run_dmop(
        method: str, 
        problem_string: str, 
        tau_t: int = 2, 
        n_t: int = 5, 
        n_var: int = 10, 
        pop_size: int = 100, 
        repetition: int = 10, 
        count_metric: bool = True, 
        save: bool = True, 
        verbose: bool = True, 
        seed: int = None, 
        device: str = 'cpu',
        suffix: str = '', 
        n_change_time: int = 20, 
): 
    if suffix != '': suffix = '_'+suffix
    # Experimental Settings
    change_frequency = tau_t
    change_severity = n_t
    max_n_gen = n_change_time * change_frequency
    termination = get_termination("n_gen", max_n_gen)

    def reset_metrics():
        global po_gen, igds, hvs, hvds, igd_all, hv_all
        po_gen = []
        igds = []
        hvs = []
        hvds = []
        igd_all = []
        hv_all = []

    def update_metrics(algorithm):
        global ind, po_gen, igds, hvs, hvds, igd_all, hv_all
        ind += 1
        _F = algorithm.opt.get("F")
        if count_metric:
            PF = algorithm.problem._calc_pareto_front(n_pareto_points=pop_size)
            igd = IGD(PF).do(_F)
            igd_all.append(igd)
            hv = HV(pf=PF).do(_F)
            hv_all.append(hv)

            if (ind % tau_t == 0): 
                igds.append(igd)
                hvs.append(hv)
                # hvds.append(
                #     Hypervolume().do(PF) - Hypervolume().do(_F)
                # )
                # po_gen.append(algorithm.opt.get("X"))

    

    class DefaultDynCallback(Callback):

        def _update(self, algorithm):

            update_metrics(algorithm)

    # Function to run an algorithm and return the results
    def run_algorithm(problem, algorithm, termination, seed):
        global ind, po_gen, igds, hvs, hvds, igd_all, hv_all
        reset_metrics()
        simulation = TimeSimulation()
        callback = CallbackCollection(DefaultDynCallback(), simulation)
        res = minimize(problem, algorithm, termination=termination, callback=callback, seed=seed, verbose=False)
        return res, igds, hvs, hvds, po_gen, igd_all, hv_all
    
    
    if count_metric: 
        POS = []
        IGDS = []
        HVS = []
        IGD_ALL = []
        HV_ALL = []
        TIMES = []
        HVDS = []

    file_path = f"results/dmop/{problem_string.lower()}/{method.lower()}_nt_{n_t}_taut_{tau_t}{suffix}.pickle"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    global ind

    match method.lower(): 
        case "dnsga2a": 
            # DNSGA2A
            for i in range(repetition): 
                ind = 1
                problem = get_problem(problem_string, taut=change_frequency, nt=change_severity, n_var=n_var)
                algorithm = DNSGA2(pop_size=pop_size, version="A")
                start = time.time()
                res, igds_tmp, hvs_tmp, hvds_tmp, po_gen_tmp, igd_all_tmp, hv_all_tmp = run_algorithm(problem, algorithm, termination, seed)
                time_tmp = time.time() - start
                IGDS.append(igds_tmp)
                HVS.append(hvs_tmp)
                POS.append(po_gen_tmp)
                TIMES.append(time_tmp)
                HVDS.append(hvds_tmp)
                IGD_ALL.append(igd_all_tmp)
                HV_ALL.append(hv_all_tmp)
                if verbose: 
                    print(f"Method: {method} | Problem: {problem_string}, tau_t: {tau_t}, n_t: {n_t} | Trial {i}: MIGD: {np.mean(igds_tmp):.2e}, MHV: {np.mean(hvs_tmp):.2e}, Evaluation Time: {time_tmp:.2f}")
        
        case "dnsga2b": 
            # DNSGA2B
            for i in range(repetition): 
                ind = 1
                problem = get_problem(problem_string, taut=change_frequency, nt=change_severity, n_var=n_var)
                algorithm = DNSGA2(pop_size=pop_size, version="B")
                start = time.time()
                res, igds_tmp, hvs_tmp, hvds_tmp, po_gen_tmp, igd_all_tmp, hv_all_tmp = run_algorithm(problem, algorithm, termination, seed)
                time_tmp = time.time() - start
                IGDS.append(igds_tmp)
                HVS.append(hvs_tmp)
                POS.append(po_gen_tmp)
                TIMES.append(time_tmp)
                HVDS.append(hvds_tmp)
                IGD_ALL.append(igd_all_tmp)
                HV_ALL.append(hv_all_tmp)
                if verbose: 
                    print(f"Method: {method} | Problem: {problem_string}, tau_t: {tau_t}, n_t: {n_t} | Trial {i}: MIGD: {np.mean(igds_tmp):.2e}, MHV: {np.mean(hvs_tmp):.2e}, Evaluation Time: {time_tmp:.2f}")

        case "ppsl-bo": 
            # PPSL-BO
            for rep in range(repetition): 
                problem = mop_dyn(pname=problem_string, n_dim=n_var, taut=change_frequency, nt=change_severity)
                start = time.time()
                out = run_dmop_ppsl_bo(
                    problem=problem, 
                    tau_t=tau_t, 
                    max_n_gen=max_n_gen, 
                    pop_size=pop_size, 
                    hpn_hidden_size=1024, 
                    psm_hidden_size=256, 
                    n_candidate=500, 
                    psm_n_layer=2, 
                    lr_hpn=.5e-5,  
                    lr_base=.2e-3, 
                    loss_type='stch', 
                    n_init=20, 
                    batch_size=8,
                    coef_lcb=.02,
                    lora_type=True, 
                    free_rank=3, 
                    device=device,             
                    verbose=verbose,        
                )
                igds, hvs = out['IGD'], out['HV']
                igd_all, hv_all = out['IGD_ALL'], out['HV_ALL']
                time_tmp = time.time() - start

                IGDS.append(igds)
                HVS.append(hvs)
                TIMES.append(time_tmp)
                IGD_ALL.append(igd_all)
                HV_ALL.append(hv_all)
                if verbose: 
                    print(f"Method: {method} | Problem: {problem_string}, tau_t: {tau_t}, n_t: {n_t} | Trial {rep}: MIGD(200): {np.mean(igds):.2e}, MHV(200): {np.mean(hvs):.2e}, Evaluation Time: {time_tmp:.2f}")


    if save: 
        with open(file_path, 'wb') as f: 
            pickle.dump(
                {"igds": IGDS, 
                "hvs": HVS, 
                "hvds": HVDS, 
                "times": TIMES,
                "igd_all": IGD_ALL, 
                "hv_all": HV_ALL,
                }, f
            )


def run_experiments(method):
    # params
    n_t_values = [5, 10, 20]
    
    # pop_size
    problems_config = {
        **{f"df{i}": 100 for i in range(1, 15)},  # df1-df9: pop_size=100
        # **{f"df{i}": 150 for i in range(10, 15)}  # df10-df14: pop_size=150
    }
    
    total_experiments = len(n_t_values) * len(problems_config)
    print(f"Total experiments to run: {total_experiments}")
    print(f"Starting at {datetime.now()}")
    
    count = 0
    for n_t in n_t_values:
        for problem, pop_size in problems_config.items():
            count += 1
            print(f"\n[{count}/{total_experiments}] Running: {problem} (tau_t={2}, n_t={n_t}, pop_size={pop_size})")
            
            try:
                run_dmop(
                    method=method,
                    problem_string=problem,
                    tau_t=2,
                    n_t=n_t,
                    pop_size=pop_size,
                    repetition=3,
                    n_change_time=20,
                    device='cpu' if method in ['dnsga2a', 'dnsga2b'] else 'cuda',
                    verbose=True,
                    save=True
                )
                print(f"✓ Completed: {problem}")
            except Exception as e:
                print(f"✗ Failed: {problem}, Error: {e}")
    
    print(f"\nAll experiments completed at {datetime.now()}")



if __name__ == "__main__":
    run_experiments(method='ppsl-bo')  # Change to 'dnsga2a' or 'dnsga2b' as needed