import os

import torch 
import numpy as np 



# ------------------------------------------------------------------------
# ---------- Parametric Multi-Objective Optimization Problem -------------
# ------------------------------------------------------------------------
from pymoo.util.nds.efficient_non_dominated_sort import efficient_non_dominated_sort

class PMOP_testA(): 
    def __init__(self, n_params, n_obj, n_dim, device: str = 'cpu') -> None:
        assert n_params >= 0, "Number of parameters must be non-negative!"
        assert n_obj > 0, "Number of objecitve must be non-negative!"
        assert n_params+n_obj <= n_dim
        self.n_params = n_params
        self.n_obj = n_obj
        self.n_dim = n_dim 
        self.device = device
        self.xl = torch.zeros([self.n_vari], device=self.device)
        self.xu = torch.ones([self.n_vari], device=self.device)
        # parameters' scales
        self.params_scale = 1. * torch.ones([self.n_params], device=self.device)
        self.params_offset = 0. * torch.ones([self.n_params], device=self.device)

    @property
    def vari_dim(self): 
        return self.n_vari
    
    def pareto_front(self, param, n_points: int = None): 
        assert param.shape[0] == self.n_params
        param = self.params_scale * param + self.params_offset
        if self.n_obj == 2: 
            f1 = torch.linspace(0, 1, 200, device=self.device) 
            f2 = 3 * (self.n_obj+self.n_params) - (2 * param + torch.sin(3*torch.pi*param)).sum() - \
                                                    (2 * f1 + torch.sin(3*torch.pi*f1))
            F = torch.hstack([f1.reshape(-1, 1), f2.reshape(-1, 1)])
            np_F = F.cpu().numpy()
            nd_F = np_F[efficient_non_dominated_sort(np_F)[0]]
            ind_1, ind_2 = nd_F[:, 0]<0.2, nd_F[:, 0]>0.2
            part_1, part_2 = nd_F[ind_1], nd_F[ind_2]
            return part_1, part_2
        elif self.n_obj == 3: 
            f1, f2 = torch.linspace(0, 1, steps=100, device=self.device), torch.linspace(0, 1, steps=100, device=self.device)
            fx, fy = torch.meshgrid(f1, f2, indexing='xy')
            fz = 3 * (self.n_obj+self.n_params) - (2 * param + torch.sin(3*torch.pi*param)).sum() - \
                                                    (2 * fx + torch.sin(3*torch.pi*fx)) - \
                                                        (2 * fy + torch.sin(3*torch.pi*fy))
            np_F = torch.hstack([fx.reshape(-1, 1), fy.reshape(-1, 1), fz.reshape(-1, 1)]).cpu().numpy()
            nd_F = np_F[efficient_non_dominated_sort(np_F)[0]]
            return nd_F
            
    def evaluate(self, x, param): 
        x = torch.atleast_2d(x) 
        assert x.shape[1] == self.n_vari
        # param = torch.atleast_2d(param)
        assert param.shape[0] == self.n_params
        param = self.params_scale * param + self.params_offset

        obj = torch.zeros([x.shape[0], self.n_obj], device=self.device)
        obj[:, :self.n_obj-1] = 1/2 * (1 + torch.sin(20*x[:,:self.n_obj-1]))
        obj[:, -1] = (1 + 9 / (6-self.n_obj-self.n_params)*torch.sum(x[:,self.n_obj-1:], dim=1)) * \
            (3*(self.n_obj+self.n_params) - (2*param+torch.sin(3*torch.pi*param)).sum()- \
                (2*obj[:, :self.n_obj-1]+torch.sin(3*torch.pi*obj[:, :self.n_obj-1])).sum(dim=1))
        return obj

    

# ------------------------------------------------------------------------
# ---------- Homotopy Multi-Objective Optimization Problem ---------------
# ------------------------------------------------------------------------
from pymoo.core.problem import ElementwiseProblem
from pymoo.problems import get_problem
p_true = get_problem("zdt4")

class zdt4_homo(ElementwiseProblem):

    def __init__(self, t):
        self.t = t
        n_t = 10000 
        self.u = np.random.randn(n_t, p_true.n_var)
        xl = p_true.xl
        xu = p_true.xu
        super().__init__(n_var=10, n_obj=2, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        n_t = self.u.shape[0]
        tmp_x = np.minimum(np.maximum(x+self.t*self.u, p_true.xl), p_true.xu)
        out["F"] = np.sum(p_true.evaluate(tmp_x), axis=0) / n_t
        


# ------------------------------------------------------------------------
# ------------------ MOP with structure constraints ----------------------
# ------------------------------------------------------------------------
from problems.problem_f_re import get_problem_f_re

class mop_sc(): 
    def __init__(self, 
                 pname: str, 
                 share_comp: list, 
                ) -> None:
        self.share_comp = share_comp 
        self.p = get_problem_f_re(pname) 
        # ideal and nadir points for RE problems
        if pname.lower() in ['re21', 're24', 're32', 're33','re36','re37']:
            self.ideal_point = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'RE/ideal_nadir_points/ideal_point_{pname.upper()}.dat'))
            self.nadir_point = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'RE/ideal_nadir_points/nadir_point_{pname.upper()}.dat'))
        else: 
            self.ideal_point = self.p.ideal_point
            self.nadir_point = self.p.nadir_point
        # bounds for variables and parameters
        mask = torch.ones(self.p.n_dim, dtype=bool)
        mask[self.share_comp] = False
        self.xl = self.p.lbound[mask] 
        self.xu = self.p.ubound[mask] 
        self.pl = self.p.lbound[~mask] 
        self.pu = self.p.ubound[~mask] 
        
    @property
    def n_dim(self): 
        return self.p.n_dim - self.n_params
    
    @property
    def n_obj(self): 
        return self.p.n_obj
    
    @property
    def n_params(self): 
        return len(self.share_comp)
    
    def normalize_p(self, param): 
        if self.pl.device.type != param.device.type: 
            device = param.device.type
            self.pu = self.pu.to(device)
            self.pl = self.pl.to(device)
        return (param - self.pl) / (self.pu - self.pl)
    
    def unnormalize_p(self, param): 
        if self.pl.device.type != param.device.type: 
            device = param.device.type
            self.pu = self.pu.to(device)
            self.pl = self.pl.to(device)
        return param * (self.pu - self.pl) + self.pl
    
    def normalize_x(self, x): 
        if self.xl.device.type != x.device.type: 
            device = x.device.type
            self.xu = self.xu.to(device)
            self.xl = self.xl.to(device)
        return (x - self.xl) / (self.xu - self.xl)
    
    def unnormalize_x(self, x): 
        if self.xl.device.type != x.device.type: 
            device = x.device.type
            self.xu = self.xu.to(device)
            self.xl = self.xl.to(device)
        return x * (self.xu - self.xl) + self.xl

    def set_share_comp(self, share_comp: list) -> None: 
        self.share_comp = share_comp
        
        mask = torch.ones(self.p.n_dim, dtype=bool)
        mask[self.share_comp] = False
        self.xl = self.p.lbound[mask] 
        self.xu = self.p.ubound[mask] 
        self.pl = self.p.lbound[~mask] 
        self.pu = self.p.ubound[~mask] 

    def evaluate(self, x, param): 
        device = x.device.type
        if x.device.type != self.xl.device.type:        
            self.xl = self.xl.to(device)
            self.xu = self.xu.to(device)
        
        # param = self.normalize_p(torch.atleast_1d(param.to(device)))
        x = torch.atleast_2d(x)
        nr = x.shape[0]
        
        for i, index in enumerate(self.share_comp): 
            x = torch.cat((x[:,:index].reshape(nr,-1), torch.ones((nr,1), device=device)*param[i], x[:,index:].reshape(nr,-1)), dim=1)

        return self.p.evaluate(x)



# ------------------------------------------------------------------------
# ------------------ Dynamic Multi-Objective Problems --------------------
# ------------------------------------------------------------------------
from .problem_dyn import get_problem_dyn

class mop_dyn(): 
    def __init__(self,
                 pname: str, 
                 n_dim: int, 
                 tau: int = 0,
                 nt: int = 10, 
                 taut: int = 20, 
                 ):
        # self.p = get_problem(pname, taut=1, nt=change_severity, n_var=n_dim)
        self.p = get_problem_dyn(pname, n_dim=n_dim)
        self.n_dim = n_dim
        self.xl, self.xu = torch.tensor(self.p.xl), torch.tensor(self.p.xu)
        self.pl, self.pu = torch.tensor([0.]), torch.tensor([1.])
        self.ideal_point = self.p.ideal_point
        self.nadir_point = self.p.nadir_point
        
        self.tau = tau
        self.nt = nt
        self.taut = taut
        
        self._time = None
    
    @property
    def n_obj(self): 
        return self.p.n_obj
    
    @property
    def n_params(self): 
        return 1
    
    @property
    def time(self):
        if self._time is not None:
            return self._time
        else:
            return 1 / self.nt * (self.tau // self.taut)

    @time.setter
    def time(self, value):
        self._time = value
    
    def normalize_p(self, param): 
        if self.pl.device.type != param.device.type: 
            device = param.device.type
            self.pu = self.pu.to(device)
            self.pl = self.pl.to(device)
        return (param - self.pl) / (self.pu - self.pl)
    
    def unnormalize_p(self, param): 
        if self.pl.device.type != param.device.type: 
            device = param.device.type
            self.pu = self.pu.to(device)
            self.pl = self.pl.to(device)
        return param * (self.pu - self.pl) + self.pl

    def tic(self, elapsed: int = 1): 
        # increase the time counter by one
        self.tau += elapsed
        
    def evaluate(self, x, param = None): 
        if self.xu.device.type != x.device.type: 
            device = x.device.type
            self.xu = self.xu.to(device)
            self.xl = self.xl.to(device)
        x = x * (self.xu - self.xl) + self.xl
        return self.p._evaluate(x, time=self.time)
    
    def _calc_pareto_front(self, n_pareto_points=100, **kwargs):
        return self.p._calc_pareto_front(time=self.time, n_pareto_points=n_pareto_points)
    


        
        
if __name__ == '__main__':
    print()
    p = get_problem_f_re('synthetic')
    p_sc = mop_sc(pname='synthetic', share_comp=[2])
    
    param = 0.5 
    
    x_part = np.random.rand(10, 2)
    x = np.hstack([x_part, np.ones(10, ).reshape(-1, 1) * 0.75])

    y = p.evaluate(torch.tensor(x))
    y_sc = p_sc.evaluate(torch.tensor(x_part), param = torch.tensor([param]))
    
    print(x)
    print(y_sc)

    import matplotlib.pyplot as plt
    fig = plt.figure()

    plt.scatter(y[:,0],y[:,1], c = 'C0',  alpha = .5)
    plt.scatter(y_sc[:,0],y_sc[:,1], c = 'C2',  alpha = .5)
    plt.show()