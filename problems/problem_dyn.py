import torch
import numpy as np

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def get_problem_dyn(name, *args, **kwargs):
    name = name.lower()
    
    PROBLEM = {
        'df1': DF1,
        'df2': DF2,
        'df3': DF3,
        'df4': DF4,
        'df5': DF5,
        'df6': DF6,
        'df7': DF7,
        'df8': DF8,
        'df9': DF9,
        'df10': DF10,
        'df11': DF11,
        'df12': DF12,
        'df13': DF13,
        'df14': DF14,
 }

    if name not in PROBLEM:
        raise Exception("Problem not found.")
    
    return PROBLEM[name](*args, **kwargs)


class DF():
    def __init__(self, n_dim: int): 
        self.n_dim = n_dim


class DF1(DF):
    
    def __init__(self, n_dim):
        super().__init__(n_dim)
        self.n_obj = 2
        
        self.xl = [0.] * self.n_dim
        self.xu = [1.] * self.n_dim
        self.ideal_point = [0., 0.]
        self.nadir_point = [1., 1.]
    
    @staticmethod
    def _evaluate(x, time, *args, **kwargs):
        device = x.device.type
        time = torch.tensor(time).to(device)
        
        v = torch.sin(0.5 * torch.pi * time)

        G = torch.abs(v)
        H = 0.75 * v + 1.25
        g = 1 + torch.sum((x[:, 1:] - G) ** 2, dim=1)

        f1 = x[:, 0]
        f2 = g * (1 - ((f1 / g) ** H))

        return torch.column_stack([f1, f2])

    @staticmethod
    def _calc_pareto_front(time, n_pareto_points=100, **kwargs):
        v = np.sin(0.5 * np.pi * time)
        H = 0.75 * v + 1.25

        f1 = np.linspace(0, 1, n_pareto_points)
        return np.array([f1, 1 - f1 ** H]).T


class DF2(DF):
    
    def __init__(self, n_dim):
        super().__init__(n_dim)
        self.n_obj = 2
        
        self.xl = [0.] * self.n_dim
        self.xu = [1.] * self.n_dim
        self.ideal_point = [0., 0.]
        self.nadir_point = [1., 1.]
    
    @staticmethod
    def _evaluate(x, time, *args, **kwargs):
        device = x.device.type
        time = torch.tensor(time).to(device)
        
        v = torch.sin(0.5 * torch.pi * time)
        G = torch.abs(v)

        n = x.shape[1]
        r = int((n - 1) * G)
        not_r = [k for k in range(n) if k != r]

        g = 1 + torch.sum((x[:, not_r] - G) ** 2, dim=1)

        f1 = x[:, r]
        f2 = g * (1 - torch.pow(f1 / g, 0.5))

        return torch.column_stack([f1, f2])
    
    @staticmethod
    def _calc_pareto_front(time, *args, n_pareto_points=100, **kwargs):
        f1 = np.linspace(0, 1, n_pareto_points)
        return np.array([f1, 1 - np.sqrt(f1)]).T
    
    
class DF3(DF):
    
    def __init__(self, n_dim):
        super().__init__(n_dim)
        self.n_obj = 2
        
        self.xl = [0.] + [-1.] * (self.n_dim - 1)
        self.xu = [1.] + [2.] * (self.n_dim - 1)
        self.ideal_point = [0., 0.]
        self.nadir_point = [1., 1.]
    
    @staticmethod
    def _evaluate(x, time, *args, **kwargs):
        device = x.device.type
        time = torch.tensor(time).to(device)
        
        G = torch.sin(0.5 * torch.pi * time)
        H = G + 1.5

        g = 1 + torch.sum((x[:, 1:] - G - x[:, [0]] ** H) ** 2, dim=1)
        f1 = x[:, 0]
        f2 = g * (1 - (x[:, 0] / g) ** H)

        return torch.column_stack([f1, f2])
    
    @staticmethod
    def _calc_pareto_front(time, *args, n_pareto_points=100, **kwargs):
        x = np.linspace(0, 1, n_pareto_points)
        g = 1

        G = np.sin(np.dot(np.dot(0.5, np.pi), time))
        H = G + 1.5
        f1 = np.copy(x)
        f2 = np.dot(g, (1 - (x / g) ** H))

        return np.column_stack([f1, f2])


class DF4(DF):
    
    def __init__(self, n_dim):
        super().__init__(n_dim)
        self.n_obj = 2
        
        self.xl = [-2.] * (self.n_dim)
        self.xu = [2.] * (self.n_dim)
        self.ideal_point = [0., 0.]
        self.nadir_point = [1., 1.]
    
    @staticmethod
    def _evaluate(x, time, *args, **kwargs):
        device = x.device.type
        time = torch.tensor(time).to(device)
        
        n = x.shape[1]
        a = torch.sin(0.5 * torch.pi * time)
        b = 1 + torch.abs(torch.cos(0.5 * torch.pi * time))
        H = 1.5 + a
        c = torch.maximum(torch.abs(a), a + b)

        g = 1.0
        for i in range(1, n):
            g += (x[:, i] - (a * (x[:, 0] / c) ** 2 / (i + 1))) ** 2

        f1 = g * torch.abs(x[:, 0] - a) ** H
        f2 = g * torch.abs(x[:, 0] - a - b) ** H

        return torch.column_stack([f1, f2])

    @staticmethod
    def _calc_pareto_front(time, *args, n_pareto_points=100, **kwargs):
        a = np.sin(0.5 * np.pi * time)
        b = 1 + np.abs(np.cos(0.5 * np.pi * time))
        H = 1.5 + a
        x = np.linspace(a, a + b, n_pareto_points)

        f1 = np.abs(x - a) ** H
        f2 = np.abs(x - a - b) ** H

        return np.array([f1, f2]).T


class DF5(DF):
    
    def __init__(self, n_dim):
        super().__init__(n_dim)
        self.n_obj = 2
        
        self.xl = [0.] + [-1.] * (self.n_dim - 1)
        self.xu = [1.] + [1.] * (self.n_dim - 1)
        self.ideal_point = [0., 0.]
        self.nadir_point = [1., 1.]
    
    @staticmethod
    def _evaluate(x, time, *args, **kwargs):
        device = x.device.type
        time = torch.tensor(time).to(device)
        
        G = torch.sin(0.5 * torch.pi * time)
        w = torch.floor(10 * G)
        g = 1 + torch.sum((x[:, 1:] - G) ** 2, dim=1)
        f1 = g * (x[:, 0] + 0.02 * torch.sin(w * torch.pi * x[:, 0]))
        f2 = g * (1 - x[:, 0] + 0.02 * torch.sin(w * torch.pi * x[:, 0]))

        return torch.column_stack([f1, f2])

    @staticmethod
    def _calc_pareto_front(time, *args, n_pareto_points=100, **kwargs):
        x = np.linspace(0, 1, n_pareto_points)
        G = np.sin(0.5 * np.pi * time)
        w = np.floor(10 * G)
        f1 = x + 0.02 * np.sin(w * np.pi * x)
        f2 = 1 - x + 0.02 * np.sin(w * np.pi * x)
        return np.array([f1, f2]).T


class DF6(DF):
    
    def __init__(self, n_dim):
        super().__init__(n_dim)
        self.n_obj = 2
        
        self.xl = [0.] + [-1.] * (self.n_dim - 1)
        self.xu = [1.] + [1.] * (self.n_dim - 1)
        self.ideal_point = [0., 0.]
        self.nadir_point = [1., 1.]
    
    @staticmethod
    def _evaluate(x, time, *args, **kwargs):
        device = x.device.type
        time = torch.tensor(time).to(device)
        
        G = torch.sin(0.5 * torch.pi * time)
        a = 0.2 + 2.8 * torch.abs(G)
        y = x[:, 1:] - G
        g = 1 + torch.sum((torch.abs(G) * y ** 2 - 10 * torch.cos(2 * torch.pi * y) + 10), axis=1)

        f1 = g * torch.pow(x[:, 0] + 0.1 * torch.sin(3 * torch.pi * x[:, 0]), a)
        f2 = g * torch.pow(1 - x[:, 0] + 0.1 * torch.sin(3 * torch.pi * x[:, 0]), a)

        return torch.column_stack([f1, f2])

    @staticmethod
    def _calc_pareto_front(time, *args, n_pareto_points=100, **kwargs):
        x = np.linspace(0, 1, n_pareto_points)
        G = np.sin(0.5 * np.pi * time)
        a = 0.2 + 2.8 * np.abs(G)
        f1 = (x + 0.1 * np.sin(3 * np.pi * x)) ** a
        f2 = (1 - x + 0.1 * np.sin(3 * np.pi * x)) ** a

        return np.array([f1, f2]).T


class DF7(DF):
    
    def __init__(self, n_dim):
        super().__init__(n_dim)
        self.n_obj = 2
        
        self.xl = [1.] + [0.] * (self.n_dim - 1)
        self.xu = [4.] + [1.] * (self.n_dim - 1)
        self.ideal_point = [0., 0.]
        self.nadir_point = [1., 1.]
    
    @staticmethod
    def _evaluate(x, time, *args, **kwargs):
        device = x.device.type
        time = torch.tensor(time).to(device)
        
        a = 5 * torch.cos(0.5 * torch.pi * time)
        g = 1 + torch.sum((x[:, 1:] - 1 / (1 + torch.exp(a * (x[:, [0]] - 2.5)))) ** 2, axis=1)

        f1 = g * (1 + time) / x[:, 0]
        f2 = g * x[:, 0] / (1 + time)

        return torch.column_stack([f1, f2])

    @staticmethod
    def _calc_pareto_front(time, *args, n_pareto_points=100, **kwargs):
        x = np.linspace(1, 4, n_pareto_points)
        f1 = (1 + time) / x
        f2 = x / (1 + time)
        pf = np.array([f1, f2]).T
        pf = pf[np.argsort(pf[:, 0])]
        return pf


class DF8(DF):
    
    def __init__(self, n_dim):
        super().__init__(n_dim)
        self.n_obj = 2
        
        self.xl = [0.] + [-1.] * (self.n_dim - 1)
        self.xu = [1.] + [1.] * (self.n_dim - 1)
        self.ideal_point = [0., 0.]
        self.nadir_point = [1., 1.]
    
    @staticmethod
    def _evaluate(x, time, *args, **kwargs):
        device = x.device.type
        time = torch.tensor(time).to(device)
        
        G = torch.sin(0.5 * torch.pi * time)
        a = 2.25 + 2 * torch.cos(2 * torch.pi * time)
        b = 100 * (G ** 2)
        tmp = G * torch.sin(4 * torch.pi * torch.pow(x[:, 0].reshape(len(x), 1), b)) / (1 + torch.abs(G))
        g = 1 + torch.sum((x[:, 1:] - tmp) ** 2, dim=1)
        f1 = g * (x[:, 0] + 0.1 * torch.sin(3 * torch.pi * x[:, 0]))
        f2 = g * torch.pow(1 - x[:, 0] + 0.1 * torch.sin(3 * torch.pi * x[:, 0]), a)

        return torch.column_stack([f1, f2])

    @staticmethod
    def _calc_pareto_front(time, *args, n_pareto_points=100, **kwargs):
        x = np.linspace(0, 1, n_pareto_points)
        a = 2.25 + 2 * np.cos(2 * np.pi * time)

        f1 = x + 0.1 * np.sin(3 * np.pi * x)
        f2 = (1 - x + 0.1 * np.sin(3 * np.pi * x)) ** a

        return np.array([f1, f2]).T


class DF9(DF):
    
    def __init__(self, n_dim):
        super().__init__(n_dim)
        self.n_obj = 2
        
        self.xl = [0.] + [-1.] * (self.n_dim - 1)
        self.xu = [1.] + [1.] * (self.n_dim - 1)
        self.ideal_point = [0., 0.]
        self.nadir_point = [1., 1.]
    
    @staticmethod
    def _evaluate(x, time, *args, **kwargs):
        device = x.device.type
        time = torch.tensor(time).to(device)
        
        _, n = x.shape
        N = 1 + torch.floor(10 * torch.abs(torch.sin(0.5 * torch.pi * time)))
        g = 1
        for i in range(1, n):
            tmp = x[:, i] - torch.cos(4 * time + x[:, 0] + x[:, i - 1])
            g = g + tmp ** 2
        f1 = g * (x[:, 0] + torch.maximum(torch.tensor(0, device=device), (0.1 + 0.5 / N) * torch.sin(2 * N * torch.pi * x[:, 0])))
        f2 = g * (1 - x[:, 0] + torch.maximum(torch.tensor(0, device=device), (0.1 + 0.5 / N) * torch.sin(2 * N * torch.pi * x[:, 0])))

        return torch.column_stack([f1, f2])

    @staticmethod
    def _calc_pareto_front(time, *args, n_pareto_points=100, **kwargs):
        x = np.linspace(0, 1, n_pareto_points)
        N = 1 + np.floor(10 * np.abs(np.sin(0.5 * np.pi * time)))

        f1 = x + np.maximum(0, (0.1 + 0.5 / N) * np.sin(2 * N * np.pi * x))
        f2 = 1 - x + np.maximum(0, (0.1 + 0.5 / N) * np.sin(2 * N * np.pi * x))

        h = get_PF(np.array([f1, f2]), True)
        return h


class DF10(DF):
    
    def __init__(self, n_dim):
        super().__init__(n_dim)
        self.n_obj = 3
        
        self.xl = [0.] * 2 + [-1.] * (self.n_dim - 2)
        self.xu = [1.] * 2 + [1.] * (self.n_dim - 2)
        self.ideal_point = [0., 0., 0.]
        self.nadir_point = [1., 1., 1.]
    
    @staticmethod
    def _evaluate(x, time, *args, **kwargs):
        device = x.device.type
        time = torch.tensor(time).to(device)
        
        G = torch.sin(0.5 * torch.pi * time)
        H = 2.25 + 2 * torch.cos(0.5 * torch.pi * time)
        x0 = x[:, 0].reshape(len(x), 1)
        x1 = x[:, 1].reshape(len(x), 1)
        tmp = torch.sin(2 * torch.pi * (x0 + x1)) / (1 + torch.abs(G))  # in the document is 2*
        g = 1 + torch.sum((x[:, 2:] - tmp) ** 2, dim=1)
        g = g.reshape(len(g), 1)
        f1 = (g * torch.pow(torch.sin(0.5 * torch.pi * x0), H)).reshape(len(g), )
        f2 = (g * torch.pow(torch.sin(0.5 * torch.pi * x1) * torch.cos(0.5 * torch.pi * x0), H)).reshape(len(g), )
        f3 = (g * torch.pow(torch.cos(0.5 * torch.pi * x1) * torch.cos(0.5 * torch.pi * x0), H)).reshape(len(g), )

        return torch.column_stack([f1, f2, f3])

    @staticmethod
    def _calc_pareto_front(time, *args, n_pareto_points=100, **kwargs):
        H = 10
        x1, x2 = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, H), indexing='xy')
        H = 2.25 + 2 * np.cos(0.5 * np.pi * time)
        g = 1
        f1 = g * np.sin(0.5 * np.pi * x1) ** H
        f2 = np.multiply(g * np.sin(0.5 * np.pi * x2) ** H, np.cos(0.5 * np.pi * x1) ** H)
        f3 = np.multiply(g * np.cos(0.5 * np.pi * x2) ** H, np.cos(0.5 * np.pi * x1) ** H)

        return get_PF(np.array([f1, f2, f3]), False)


class DF11(DF):
    
    def __init__(self, n_dim):
        super().__init__(n_dim)
        self.n_obj = 3
        
        self.xl = [0.] * self.n_dim
        self.xu = [1.] * self.n_dim
        self.ideal_point = [0., 0., 0.]
        self.nadir_point = [1., 1., 1.]
    
    @staticmethod
    def _evaluate(x, time, *args, **kwargs):
        device = x.device.type
        time = torch.tensor(time).to(device)
        
        G = torch.abs(torch.sin(0.5 * torch.pi * time))
        g = 1 + G + torch.sum((x[:, 2:] - 0.5 * G * x[:, 0].reshape(len(x), 1)) ** 2, dim=1)
        y = [torch.pi * G / 6.0 + (torch.pi / 2 - torch.pi * G / 3.0) * x[:, i] for i in [0, 1]]

        f1 = g * torch.sin(y[0])
        f2 = g * torch.sin(y[1]) * torch.cos(y[0])
        f3 = g * torch.cos(y[1]) * torch.cos(y[0])

        return torch.column_stack([f1, f2, f3])

    @staticmethod
    def _calc_pareto_front(time, *args, n_pareto_points=100, **kwargs):
        H = 20
        x1, x2 = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, H), indexing='xy')
        G = np.abs(np.sin(0.5 * np.pi * time))
        # g = 1 + G + np.sum((x[:, 2:] - 0.5 * G * x[:, 0].reshape(len(x), 1)) ** 2, axis=1)
        y1 = np.pi * G / 6 + (np.pi / 2 - np.pi * G / 3) * x1
        y2 = np.pi * G / 6 + (np.pi / 2 - np.pi * G / 3) * x2

        f1 = np.sin(y1)
        f2 = (1+G) * np.sin(y2) * np.cos(y1)
        f3 = (1+G) * np.cos(y2) * np.cos(y1)

        return get_PF(np.array([f1, f2, f3]), False)


class DF12(DF):
    
    def __init__(self, n_dim):
        super().__init__(n_dim)
        self.n_obj = 3
        
        self.xl = [0.] * 2 + [-1.] * (self.n_dim - 2)
        self.xu = [1.] * 2 + [1.] * (self.n_dim - 2)
        self.ideal_point = [0., 0., 0.]
        self.nadir_point = [1., 1., 1.]
    
    @staticmethod
    def _evaluate(x, time, *args, **kwargs):
        device = x.device.type
        time = torch.tensor(time).to(device)
        
        k = 10 * torch.sin(torch.pi * time)
        #        r = 1 - torch.modulo(k,2)
        r = 1
        x0 = x[:, 0].reshape(len(x), 1)
        tmp1 = x[:, 2:] - torch.sin(time * x0)
        tmp2 = torch.abs(torch.sin(torch.floor(k * (2 * x[:, 0:2] - r)) * torch.pi / 2))
        g = 1 + torch.sum(tmp1 ** 2, axis=1) + torch.prod(tmp2)

        f1 = g * torch.cos(0.5 * torch.pi * x[:, 1]) * torch.cos(0.5 * torch.pi * x[:, 0])
        f2 = g * torch.sin(0.5 * torch.pi * x[:, 1]) * torch.cos(0.5 * torch.pi * x[:, 0])
        f3 = g * torch.sin(0.5 * torch.pi * x[:, 0])

        return torch.column_stack([f1, f2, f3])

    @staticmethod
    def _calc_pareto_front(time, *args, n_pareto_points=100, **kwargs):
        H = 20
        x1, x2 = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, H), indexing='xy')
        k = 10 * np.sin(np.pi * time)
        tmp2 = np.abs(
            (np.sin((np.floor(k * (2 * x1 - 1)) * np.pi) / 2) *
             np.sin((np.floor(k * (2 * x2 - 1)) * np.pi) / 2)))
        g = 1 + tmp2
        f1 = np.multiply(np.multiply(g, np.cos(0.5 * np.pi * x2)), np.cos(0.5 * np.pi * x1))
        f2 = np.multiply(np.multiply(g, np.sin(0.5 * np.pi * x2)), np.cos(0.5 * np.pi * x1))
        f3 = np.multiply(g, np.sin(0.5 * np.pi * x1))

        return get_PF(np.array([f1, f2, f3]), True)


class DF13(DF):
    
    def __init__(self, n_dim):
        super().__init__(n_dim)
        self.n_obj = 3
        
        self.xl = [0.] * 2 + [-1.] * (self.n_dim - 2)
        self.xu = [1.] * 2 + [1.] * (self.n_dim - 2)
        self.ideal_point = [0., 0., 0.]
        self.nadir_point = [1., 1., 1.]
    
    @staticmethod
    def _evaluate(x, time, *args, **kwargs):
        device = x.device.type
        time = torch.tensor(time).to(device)
        
        G = torch.sin(0.5 * torch.pi * time)
        p = torch.floor(6 * G)
        x0 = x[:, 0].reshape(len(x), 1)
        x1 = x[:, 1].reshape(len(x), 1)
        g = 1 + torch.sum((x[:, 2:] - G) ** 2, dim=1)
        g = g.reshape(len(g), 1)
        f1 = g * torch.cos(0.5 * torch.pi * x0) ** 2
        f2 = g * torch.cos(0.5 * torch.pi * x1) ** 2
        f3 = g * torch.sin(0.5 * torch.pi * x0) ** 2 + torch.sin(0.5 * torch.pi * x0) * torch.cos(p * torch.pi * x0) ** 2 + torch.sin(
            0.5 * torch.pi * x1) ** 2 + torch.sin(0.5 * torch.pi * x1) * torch.cos(p * torch.pi * x1) ** 2

        return torch.column_stack([f1, f2, f3])

    @staticmethod
    def _calc_pareto_front(time, *args, n_pareto_points=100, **kwargs):
        H = 20
        x1, x2 = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, H), indexing='xy')
        G = np.sin(0.5 * np.pi * time)
        p = np.floor(6 * G)

        f1 = np.cos(0.5 * np.pi * x1) ** 2
        f2 = np.cos(0.5 * np.pi * x2) ** 2
        f3 = np.sin(0.5 * np.pi * x1) ** 2 + np.sin(0.5 * np.pi * x1) * np.cos(p * np.pi * x1) ** 2 + np.sin(
            0.5 * np.pi * x2) ** 2 + \
             np.sin(0.5 * np.pi * x2) * np.cos(p * np.pi * x2) ** 2

        return get_PF(np.array([f1, f2, f3]), True)


class DF14(DF):
    
    def __init__(self, n_dim):
        super().__init__(n_dim)
        self.n_obj = 3
        
        self.xl = [0.] * 2 + [-1.] * (self.n_dim - 2)
        self.xu = [1.] * 2 + [1.] * (self.n_dim - 2)
        self.ideal_point = [0., 0., 0.]
        self.nadir_point = [1., 1., 1.]
    
    @staticmethod
    def _evaluate(x, time, *args, **kwargs):
        device = x.device.type
        time = torch.tensor(time).to(device)
        
        G = torch.sin(0.5 * torch.pi * time)
        x0 = x[:, 0].reshape(len(x), 1)
        x1 = x[:, 1].reshape(len(x), 1)

        g = 1 + torch.sum((x[:, 2:] - G) ** 2, axis=1)
        g = g.reshape(len(g), 1)
        y = 0.5 + G * (x0 - 0.5)

        f1 = g * (1 - y + 0.05 * torch.sin(6 * torch.pi * y))
        f2 = g * (1 - x1 + 0.05 * torch.sin(6 * torch.pi * x1)) * (y + 0.05 * torch.sin(6 * torch.pi * y))
        f3 = g * (x1 + 0.05 * torch.sin(6 * torch.pi * x1)) * (y + 0.05 * torch.sin(6 * torch.pi * y))

        return torch.column_stack([f1, f2, f3])

    @staticmethod
    def _calc_pareto_front(time, *args, n_pareto_points=100, **kwargs):
        H = 20
        x1, x2 = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, H), indexing='xy')
        G = np.sin(0.5 * np.pi * time)
        y = 0.5 + G * (x1 - 0.5)
        f1 = 1 - y + 0.05 * np.sin(6 * np.pi * y)
        f2 = np.multiply(1 - x2 + 0.05 * np.sin(6 * np.pi * x2), y + 0.05 * np.sin(6 * np.pi * y))
        f3 = np.multiply(x2 + 0.05 * np.sin(6 * np.pi * x2), y + 0.05 * np.sin(6 * np.pi * y))

        return get_PF(np.array([f1, f2, f3]), False)


def get_PF(f=None, nondominate=None):
    nds = NonDominatedSorting()
    ncell = len(f)
    s = np.size(f[1])
    h = []
    for i in np.arange(ncell):
        fi = np.reshape(f[i], s, order='F')
        h.append(fi)
    h = np.array(h).T
    h = np.reshape(h, (s, ncell))

    if nondominate:
        fronts = nds.do(F=h, only_non_dominated_front=True)
        h = h[fronts]
    return h
