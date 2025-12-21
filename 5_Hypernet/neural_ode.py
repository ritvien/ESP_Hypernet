import torch
from torch import nn, Tensor
import math
import numpy as np

def ode_solve_origin(z0, t0, t1, f, fixed_n_steps=10, h_max=0.00001):
    """
    Simplest RK4 ODE initial value solver
    """
    n_steps = math.ceil((abs(t1 - t0)/h_max).max().item())
    # n_steps = 100
    h = (t1 - t0)/n_steps
    t = t0
    z = z0
#     print(n_steps)
    # h = torch.tensor([0.0001])
    # print(h)
    if fixed_n_steps != 0:
        n_steps = fixed_n_steps
    for i_step in range(n_steps):
        k1 = h * f(z,t)
        k2 = h * (f((z+h/2),t))
        k3 = h * (f((z+h/2),t))
        k4 = h * (f((z+h),t))
        k = (1/6)*(k1+2*k2+2*k3+k4)
        z = z + k      
    return z

def ode_solve_rk4(z0, t0, t1, f, fixed_n_steps=10, h_max=0.00001):
    """
    Simplest RK4 ODE initial value solver
    """
    
    n_steps = math.ceil((abs(t1 - t0)/h_max).max().item())
    # n_steps = 100
    h = (t1 - t0)/n_steps
    t = t0
    z = z0
#     print(n_steps)
    # h = torch.tensor([0.0001])
    # print(h)
    if fixed_n_steps != 0:
        n_steps = fixed_n_steps
    for _ in range(n_steps):
        k1 = f(z, t)
        k2 = f(z + 0.5 * h * k1, t + 0.5 * h)
        k3 = f(z + 0.5 * h * k2, t + 0.5 * h)
        k4 = f(z + h * k3, t + h)
        z = z + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        t = t + h     
    return z

    
class NeuralODE(nn.Module):
    def __init__(self, func, solver_name, **solver_kwargs):
        super(NeuralODE, self).__init__()
        assert isinstance(func, ODEF)
        
        self.func = func
        self.solver_kwargs = solver_kwargs
        
        if solver_name == 'origin':
            self.ode_solver = ode_solve_origin
        elif solver_name == 'rk4':
            self.ode_solver = ode_solve_rk4
        else:
            raise ValueError(f"Solver không hợp lệ: '{solver_name}'. Vui lòng chọn 'origin' hoặc 'rk4'.")

    def forward(self, z0, t=Tensor([0., 1.]), return_whole_sequence=False):
        t = t.to(z0)
        z = ODEAdjoint.apply(z0, t, self.func.flatten_parameters(), self.func, self.ode_solver, self.solver_kwargs)
        if return_whole_sequence:
            return z
        else:
            return z[-1]

class ODEAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z0, t, flat_parameters, func, ode_solve, solver_kwargs):
        assert isinstance(func, ODEF)
        bs, *z_shape = z0.size()
        time_len = t.size(0)

        with torch.no_grad():
            z = torch.zeros(time_len, bs, *z_shape).to(z0)
            z[0] = z0
            # print("check z_0:", z0.shape)
            for i_t in range(time_len - 1):
                z0 = ode_solve(z0, t[i_t], t[i_t+1], func,  **solver_kwargs)
                # print(z0)
                z[i_t+1] = z0

        ctx.func = func
        ctx.ode_solve = ode_solve
        ctx.solver_kwargs = solver_kwargs
        ctx.save_for_backward(t, z.clone(), flat_parameters)
        return z

    @staticmethod
    def backward(ctx, dLdz):
        """
        dLdz shape: time_len, batch_size, *z_shape
        """
        func = ctx.func
        ode_solve = ctx.ode_solve
        solver_kwargs = ctx.solver_kwargs
        t, z, flat_parameters = ctx.saved_tensors
        time_len, bs, *z_shape = z.size()
        n_dim = np.prod(z_shape)
        n_params = flat_parameters.size(0)

        # Dynamics of augmented system to be calculated backwards in time
        def augmented_dynamics(aug_z_i, t_i):
            """
            tensors here are temporal slices
            t_i - is tensor with size: bs, 1
            aug_z_i - is tensor with size: bs, n_dim*2 + n_params + 1
            """
            z_i, a = aug_z_i[:, :n_dim], aug_z_i[:, n_dim:2*n_dim]  # ignore parameters and time

            # Unflatten z and a
            z_i = z_i.view(bs, *z_shape)
            a = a.view(bs, *z_shape)
            with torch.set_grad_enabled(True):
                t_i = t_i.detach().requires_grad_(True)
                z_i = z_i.detach().requires_grad_(True)
                func_eval, adfdz, adfdt, adfdp = func.forward_with_grad(z_i, t_i, grad_outputs=a)  # bs, *z_shape
                adfdz = adfdz.to(z_i) if adfdz is not None else torch.zeros(bs, *z_shape).to(z_i)
                adfdp = adfdp.to(z_i) if adfdp is not None else torch.zeros(bs, n_params).to(z_i)
                adfdt = adfdt.to(z_i) if adfdt is not None else torch.zeros(bs, 1).to(z_i)

            # Flatten f and adfdz
            func_eval = func_eval.view(bs, n_dim)
            adfdz = adfdz.view(bs, n_dim) 
            return torch.cat((func_eval, -adfdz, -adfdp, -adfdt), dim=1)

        dLdz = dLdz.view(time_len, bs, n_dim)  # flatten dLdz for convenience
        with torch.no_grad():
            ## Create placeholders for output gradients
            # Prev computed backwards adjoints to be adjusted by direct gradients
            adj_z = torch.zeros(bs, n_dim).to(dLdz)
            adj_p = torch.zeros(bs, n_params).to(dLdz)
            # In contrast to z and p we need to return gradients for all times
            adj_t = torch.zeros(time_len, bs, 1).to(dLdz)

            for i_t in range(time_len-1, 0, -1):
                z_i = z[i_t]
                t_i = t[i_t]
                f_i = func(z_i, t_i).view(bs, n_dim)

                # Compute direct gradients
                dLdz_i = dLdz[i_t]
                dLdt_i = torch.bmm(torch.transpose(dLdz_i.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0]

                # Adjusting adjoints with direct gradients
                adj_z += dLdz_i
                adj_t[i_t] = adj_t[i_t] - dLdt_i

                # Pack augmented variable
                aug_z = torch.cat((z_i.view(bs, n_dim), adj_z, torch.zeros(bs, n_params).to(z), adj_t[i_t]), dim=-1)

                # Solve augmented system backwards
                aug_ans = ode_solve(aug_z, t_i, t[i_t-1], augmented_dynamics,  **solver_kwargs)

                # Unpack solved backwards augmented system
                adj_z[:] = aug_ans[:, n_dim:2*n_dim]
                adj_p[:] += aug_ans[:, 2*n_dim:2*n_dim + n_params]
                adj_t[i_t-1] = aug_ans[:, 2*n_dim + n_params:]

                del aug_z, aug_ans

            ## Adjust 0 time adjoint with direct gradients
            # Compute direct gradients 
            dLdz_0 = dLdz[0]
            dLdt_0 = torch.bmm(torch.transpose(dLdz_0.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0]

            # Adjust adjoints
            adj_z += dLdz_0
            adj_t[0] = adj_t[0] - dLdt_0
        return adj_z.view(bs, *z_shape), adj_t, adj_p, None, None, None
        
        
class ODEF(nn.Module):
    def forward_with_grad(self, z, t, grad_outputs):
        """Compute f and a df/dz, a df/dp, a df/dt"""
        batch_size = z.shape[0]

        out = self.forward(z, t)

        a = grad_outputs
        adfdz, adfdt, *adfdp = torch.autograd.grad(
            (out,),
            (z, t) + tuple(self.parameters()),
            grad_outputs=(a),
            allow_unused=True,
            retain_graph=True
        )
        # grad method automatically sums gradients for batch items, we have to expand them back 
        if adfdp is not None:
            adfdp = torch.cat([p_grad.flatten() for p_grad in adfdp]).unsqueeze(0)
            adfdp = adfdp.expand(batch_size, -1) / batch_size
        if adfdt is not None:
            adfdt = adfdt.expand(batch_size, 1) / batch_size
        return out, adfdz, adfdt, adfdp

    def flatten_parameters(self):
        p_shapes = []
        flat_parameters = []
        for p in self.parameters():
            p_shapes.append(p.size())
            flat_parameters.append(p.flatten())
        return torch.cat(flat_parameters)
