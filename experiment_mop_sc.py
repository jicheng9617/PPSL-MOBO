import os
import numpy as np 
import torch 

import logging 
import time
import pickle

from pymoo.indicators.hv import HV
from pymoo.core.problem import Problem 
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from problems import mop_sc
from baselines_mobo import psl_mobo, mobo_botorch
from trainer import trainer_ppsl_bo_random, generate_ps

device = 'cuda' if torch.cuda.is_available() else 'cpu'
import warnings
warnings.filterwarnings("ignore", message="To copy construct from a tensor")


class pmop_pymoo(Problem): 
    def __init__(self, problem, param, device):
        self.p = problem 
        self.param = param.to(device)
        super().__init__(n_var=self.p.n_dim, n_obj=self.p.n_obj, n_ieq_constr=0, xl=np.zeros(self.p.n_dim), xu=np.ones(self.p.n_dim))

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.p.evaluate(torch.tensor(x).to(device), self.param).cpu().numpy()


class pmop_psl(): 
    def __init__(self, problem, param):
        self.p = problem 
        self.n_dim = self.p.n_dim
        self.n_obj = self.p.n_obj 
        self.ideal_point = self.p.ideal_point
        self.nadir_point = self.p.nadir_point
        self.param = param  
    
    def evaluate(self, x): 
        return self.p.evaluate(x, self.param)
    

class BotorchProblemAdapter:
    """Adapter so that each param becomes a separate BoTorch 'problem' instance."""
    def __init__(self, p, param, device):
        self.p = p
        self.param = param.to(device)
        self.n_var = p.n_dim
        self.n_obj = p.n_obj
        self.bounds = torch.stack([torch.zeros(self.n_var), torch.ones(self.n_var)])  # assumes [0,1]^d
        # ref_point: use normalized nadir point (as in your HV call)
        self.ideal_point = p.ideal_point
        self.nadir_point = p.nadir_point
        self.ref_point = [-1.1] * self.n_obj  # normalized, as in your HV

    def evaluate(self, X):
        # X: (N, n_var) torch or np array in [0,1]
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, device=self.bounds.device, dtype=torch.float32)
        Y = self.p.evaluate(X.to(self.bounds.device), self.param).cpu().numpy()
        Y_norm = (Y - self.ideal_point) / (self.nadir_point - self.ideal_point)
        # Negate for maximization (since we want to minimize)
        return -Y_norm
    

def run_botorch_mobo(problem, n_iter, n_sample, n_init, algo, device='cpu', seed=0):
    """Run BoTorch MOBO (qEHVI or qNParEGO) for one parameter instance."""
    from botorch.utils.multi_objective.pareto import is_non_dominated
    # Use the mobo_botorch function described previously
    hvs, Y_hist = mobo_botorch(
        problem=problem,
        n_iter=n_iter,
        n_sample=n_sample,
        n_init=n_init,
        algo=algo,
        dtype=torch.double,
        device=torch.device(device),
        seed=seed,
    )
    # For reporting, use final batch's non-dominated points to compute HV
    Y_final = Y_hist  # shape: (N, n_obj), already normalized
    mask = is_non_dominated(torch.tensor(Y_final, dtype=torch.double))
    Y_nd = Y_final[mask.cpu().numpy()]
    # Use pymoo's HV for consistency with your pipeline
    from pymoo.indicators.hv import HV
    hv = HV(ref_point=-np.array(problem.ref_point))
    hv_value = hv(-Y_nd.cpu().numpy())
    return hv_value


def run_mopsc(
    problem_name: str, 
    shared_comp: list, 
    hpn_hidden_size: int,
    psm_hidden_size: int, 
    n_hidden_layer: int, 
    free_rank: int, 
    loss_type: str, 
    n_repetition: int, 
    method: str, 
    n_sample_params_test: int = 10, 
    lr_hpn: float = 5e-5, 
    lr_base: float = 50e-5, 
    save_name: str = None, 
    device: str = 'cuda', 
): 
    print(f"---------------Problem {problem_name}--------------------")
    res = {}
    ## define the problem
    p = mop_sc(pname=problem_name, share_comp=shared_comp) 
    # pf = np.loadtxt("problems/RE/ParetoFront/"+problem_name+".dat")

    print(f"---------------Share {p.share_comp}--------------------")

    file_path = f"results/mopsc/{problem_name.lower()}/test_cases_{''.join(map(str, shared_comp))}.pickle"
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            params = pickle.load(f)
    else:
        params = torch.rand((n_sample_params_test, p.n_params)).to(device)
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True) 
        with open(file_path, 'wb') as f:
            pickle.dump(params, f)
 
    # ref_point = p.nadir_point
    # ref_point = [1.1*x for x in ref_point]
    hv = HV(ref_point=np.array([1.1] * p.n_obj))
    
    ## PPSL-MOBO
    if method.lower() == 'ppsl-mobo': 
        t0_train = 0 
        t0_infer_s = 0
        hv_ppsl_s = []
        print(f"----------------------PPSL-MOBO------------------------")
        for i in range(n_repetition): 
            t0_start = time.time()
            hpnet, psmodel, _ = trainer_ppsl_bo_random(
                problem=p, 
                hpn_hidden_size=hpn_hidden_size, 
                psm_hidden_size=psm_hidden_size, 
                psm_n_layer=n_hidden_layer, 
                n_init=50, 
                n_iter=30, 
                batch_size=5, 
                lr_base=lr_base, 
                lr_hpn=lr_hpn, 
                loss_type=loss_type, 
                device=device, 
                n_sample_params = 10, # * p.n_params
                n_sample_pref=10,
                n_step=50,
                lora_type=True, 
                free_rank=free_rank,
            )
            t0_end = time.time()
            t0_train += (t0_end - t0_start) 
            
            hv_value_s, hv_value_l = [], []
            # calculate and report the hypervolume value
            for param in params: 
                hpnet.eval()
                generated_ps, generated_pf = [], []
                
                t0_infer_start = time.time()
                generated_ps_s = generate_ps(p, param, hpnet, psmodel, 1000, device)
                t0_infer_end = time.time()
                t0_infer_s += (t0_infer_end - t0_infer_start)
            
                obj_s = p.evaluate(generated_ps_s, param).cpu().numpy()
                
                generated_pf_s = (obj_s - p.ideal_point) / (p.nadir_point - p.ideal_point) 
                
                hv_value_s.append(hv(generated_pf_s))
            
                print(f"Repetition {i} --> Para: {param.round(decimals=1)}, HV (small): {hv_value_s[-1]:.4f}.")
                
            hv_ppsl_s.append(hv_value_s)
        
        res['ppsl-mobo_hv'] = hv_ppsl_s
        res['ppsl-mobo_training_time'] = t0_train / n_repetition
        res['ppsl-mobo_inference_time'] = t0_infer_s / n_repetition 
    

    ## no LoRA 
    # if run_noLora:
        # n_sample_params = 5 if p.n_obj == 2 else 8
        # t1_train = 0 
        # t1_infer = 0
        # hv_ppsl_nolora = []
        # print(f"----------------------PPSL (no LoRA)------------------------")
        # for i in range(n_repetition): 
        #     t1_start = time.time()
        #     hpnet, psmodel = trainer_ppsl_random(
        #         problem=p, 
        #         hpn_hidden_size=hpn_hidden_size, 
        #         psm_hidden_size=psm_hidden_size, 
        #         psm_n_layer=n_hidden_layer, 
        #         n_epochs=1000, 
        #         lr_hpn=lr_hpn, 
        #         loss_type=loss_type, 
        #         device=device, 
        #         n_sample_pref=10,
        #         n_sample_params=n_sample_params,
        #         lora_type=False, 
        #         free_rank=free_rank,
        #     )
        #     t1_end = time.time()
        #     t1_train += (t1_end - t1_start) 
            
        #     hv_value_nolora = []
        #     # calculate and report the hypervolume value
        #     for param in params: 
        #         hpnet.eval()
        #         generated_ps, generated_pf = [], []
                
        #         t1_infer_start = time.time()
        #         generated_ps = generate_ps(p, param, hpnet, psmodel, 1000, device)
        #         t1_infer_end = time.time()
        #         t1_infer += (t1_infer_end - t1_infer_start)
            
        #         obj = p.evaluate(generated_ps, param).cpu().numpy()
                
        #         generated_pf = (obj - p.ideal_point) / (p.nadir_point - p.ideal_point) 
                
        #         hv_value_nolora.append(hv(generated_pf))
            
        #         print(f"Repetition {i} --> Para: {param.round(decimals=1)}, HV: {hv_value_nolora[-1]:.4f}.")
                
        #     hv_ppsl_nolora.append(hv_value_nolora)
        
        # res['ppsl_nolora_hv'] = hv_ppsl_nolora
        # res['ppsl_nolora_training_time'] = t1_train / n_repetition
        # res['ppsl_nolora_inference_time'] = t1_infer / n_repetition 
    

    ## PSL
    if method.lower() == 'psl-mobo': 
        print(f"----------------------PSL-MOBO------------------------")
        t_psl_train, t_psl_infer = 0., 0.
        hv_psl = [] 
        for i in range(n_repetition): 
            hv_value = []
            for param in params: 
                p_psl = pmop_psl(p, param)
                t0_start = time.time()
                psl_model, hv_psl_values = psl_mobo(
                    problem=p_psl, 
                    n_init=20, 
                    n_steps=1000, 
                    n_iter=16, 
                    n_sample=5,
                    device=device, 
                    n_pref_update=10,
                )
                t0_end = time.time()
                t_psl_train += (t0_end - t0_start) 
                
                # calculate and report the hypervolume value
                psl_model.eval()
                generated_ps, generated_pf = [], []
                
                t0_infer_start = time.time()
                with torch.no_grad():
                    alpha = torch.ones(p.n_obj, device=device)
                    pref = torch.distributions.Dirichlet(alpha).sample((1000,))
                    generated_ps = psl_model(pref)
                t0_infer_end = time.time()
                t_psl_infer += (t0_infer_end - t0_infer_start)
            
                obj = p.evaluate(generated_ps, param).cpu().numpy()
                
                generated_pf = (obj - p.ideal_point) / (p.nadir_point - p.ideal_point) 
                
                hv_value.append(hv(generated_pf))
            
                print(f"Repetition {i} --> Para: {param.round(decimals=1)}, HV: {hv_value[-1]:.4f}.")
                
            hv_psl.append(hv_value)
    
        res['psl-mobo_hv'] = hv_psl
        res['psl-mobo_training_time'] = t_psl_train / n_repetition
        res['psl-mobo_inference_time'] = t_psl_infer / n_repetition 
    

    ## MOEAs
    if method.lower() == 'moea': 
        print(f"----------------------MOEAs------------------------")
        n_evals = 100
        
        hv_nsga2 = []
        t_nsga2 = 0.
        for i in range(n_repetition): 
            hv_value_moead, hv_value_nsga2, hv_value_nsga3 = [], [], []
            for param in params: 
                p_pymoo = pmop_pymoo(p, param, device=device)
                # NSGA-2
                t_nsga2_start = time.time()
                algorithm = NSGA2(pop_size=20)
                opti_nsga2 = minimize(p_pymoo, algorithm, termination=("n_evals", n_evals), verbose=False)
                t_nsga2_end = time.time()
                t_nsga2 += (t_nsga2_end - t_nsga2_start)
                
                nsga2_pf = (opti_nsga2.F - p.ideal_point) / (p.nadir_point - p.ideal_point)
                hv_value_nsga2.append(hv(nsga2_pf))
                print(f"Repetition {i} **NSGA-2** --> Para: {param.round(decimals=1)}, HV: {hv_value_nsga2[-1]:.4f}.")
                
            hv_nsga2.append(hv_value_nsga2)
        
        res['nsga2_hv'] = hv_nsga2
        res['nsga2_time'] = t_nsga2 / n_repetition
    

    ## MOBOs
    if method.lower() in ['qehvi', 'qnparego']:
        print(f"----------------------MOBOs (BoTorch)------------------------")
        n_iter = 16  # or whatever you want
        n_init = 20
        n_sample = 5
        mobo_method = 'qehvi' if method.lower() == 'qehvi' else 'qnparego'  # or set via argument

        hv_mobo = []
        t_mobo = 0.
        for i in range(n_repetition):
            hv_value_mobo = []
            for param in params:
                # Wrap the current parameter as a BoTorch-compatible problem
                botorch_problem = BotorchProblemAdapter(p, param, device=device)
                t_mobo_start = time.time()
                hv_val = run_botorch_mobo(
                    problem=botorch_problem,
                    n_iter=n_iter,
                    n_sample=n_sample,
                    n_init=n_init,
                    algo=mobo_method,
                    device=device,
                    seed=i,  # different seed per repetition
                )
                t_mobo_end = time.time()
                t_mobo += (t_mobo_end - t_mobo_start)
                hv_value_mobo.append(hv_val)
                print(f"Repetition {i} --> Para: {param.round(decimals=1)}, HV (MOBO-{mobo_method}): {hv_val:.4f}.")
            hv_mobo.append(hv_value_mobo)
        res[f'{mobo_method}_hv'] = hv_mobo
        res[f'{mobo_method}_time'] = t_mobo / n_repetition

    
    ## save results
    if save_name is not None: 
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        with open(save_name+'.pickle', 'wb') as f: 
            pickle.dump(res, f)



if __name__ == "__main__": 
    print() 
    # 'RE21', 'RE37', 'RE33', 'RE36'
    problem = ['RE33']
    share = [[0],[1],[2],[3],[0,1],[1,2],[2,3],[0,1,2],[1,2,3]] 
    # psm_sizes = [128, 256, 512]
    # n_layers = [2, 3, 4]
    # rank_sizes = [2, 3, 5]

    # 'moea', 'qehvi', 'qnparego', 'psl-mobo', 'ppsl-mobo'
    for method in ['psl-mobo']:
        method = method.upper()
        device = 'cuda' if method.lower() in ['psl-mobo', 'ppsl-mobo'] else 'cpu'


        for a in problem: 
            for b in share: 
                save_name = f"./results/mopsc/{a.lower()}/{method}_shared_{''.join(map(str, b))}"
                # save_name = f"./results/mopsc/previous/{'LoRAorNot'}_{a}_{''.join(map(str, b))}_hpn1024_psm512layer1_r{3}_stch"
                run_mopsc(
                    problem_name=a, 
                    shared_comp=b, 
                    hpn_hidden_size=512, 
                    psm_hidden_size=256, 
                    n_hidden_layer=3, 
                    loss_type='stch', 
                    n_repetition=3, 
                    method=method, 
                    n_sample_params_test=10, 
                    lr_hpn=1e-5, 
                    lr_base=10e-4, 
                    free_rank=3, 
                    save_name=save_name, 
                    device=device,
                )