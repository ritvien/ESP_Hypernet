from tqdm import tqdm
import torch
import numpy as np
import random
import time

from hypernet import Hypernet_trans
from nnode import NNODEF
from neural_ode import NeuralODE
from naliu import TestODEF
import autograd.numpy as np1
import math
from autograd import grad
from torch import Tensor
import torch.optim.lr_scheduler as lr_scheduler

def generate_random_partition_rays(num_rays=25):
    rays = []
    
    total_angle = np.pi / 2
    step = total_angle / num_rays
    for i in range(num_rays):
        start_angle = i * step
        end_angle = (i + 1) * step
        theta = np.random.uniform(start_angle, end_angle)
        r1 = math.cos(theta)
        r2 = math.sin(theta)
        if abs(r1) < 1e-9: r1 = 0.0
        if abs(r2) < 1e-9: r2 = 0.0
            
        r = [r1, r2]
        rays.append(r)
        
    return np.array(rays)


def train_hypernetwork(problem, ref_point, num_rays=25):
    print("Training Hypernetwork (Gradient Accumulation Mode)...")
    
    # Khởi tạo model và optimizer
    hnet = Hypernet_trans(ray_hidden_dim=16, out_dim=2, n_tasks=2) 
    optimizer = torch.optim.Adam(hnet.parameters(), lr=0.001)
    
    all_rays_history = []
    
    for epoch in tqdm(range(20000)):
        hnet.train()
        optimizer.zero_grad()
        
        rays_np = generate_random_partition_rays(num_rays)
        all_rays_history.append(rays_np)
        rays_tensor = torch.from_numpy(rays_np).float()
        
        total_epoch_loss = 0.0
        
        for i in range(num_rays):
            r_vec = rays_tensor[i].unsqueeze(0) 
            
            # Forward pass
            output = hnet(r_vec).squeeze()
            
            # 1. Tính giá trị từng hàm thành phần (f1, f2...)
            vals = [func(output) for func in problem.f]
            
            val_f_tensor = torch.stack(vals)
            
            # --------------------------------
            
            # Tính Chebyshev Loss: max(r1*f1, r2*f2)
            weighted_objs = r_vec[0] * (val_f_tensor - ref_point)
            
            loss_i = torch.max(weighted_objs)
            total_epoch_loss += loss_i

        mean_loss = total_epoch_loss / num_rays
        
        if epoch % 100 == 0:
            print(f"\nEpoch {epoch}, Mean Loss: {mean_loss.item():.4f}")
            
        mean_loss.backward()
        optimizer.step()

    torch.save(hnet.state_dict(), "model/hnet_bnk.pth")
    print("Hypernetwork training complete.")
    all_rays_flat = np.vstack(all_rays_history)
    
    return hnet, all_rays_flat


def l1(x):
    return 4*x[:,:,0]**2 + 4*x[:,:,1]**2 
def l2(x):
    return (x[:,:,0]-5)**2 + (x[:,:,1]-5)**2 

def train_neural_ode(hnet, problem, ray_train, test_rays, solver_name='rk4', **solver_kwargs):
    print("Training Neural ODE...")
    sol = []
    alpha_r = 0.6
    best_err = 1000
    patience=20
    epochs_no_improve = 0
    
    func = NNODEF(4, 200, time_invariant=True)
    ode_trained = NeuralODE(func, solver_name, **solver_kwargs)
    optimizer = torch.optim.Adam(ode_trained.parameters(), lr = 0.01) 
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=50)
    def s(x,r):
        return np1.max(np.array([r[0]*problem.f1(x),r[1]*problem.f2(x)]))
    def create_dt(hnet,batch,ray_train):
        hnet.eval()
        # Flatten the list of tensors into a single list
        flattened_list = [item for sublist in ray_train for item in sublist]

        # Randomly choose 16 elements from the flattened list
        ray_in = random.sample(flattened_list, batch)

        x0 = []
        x_target = []
        t_max = 4
        n_points = 10

        index_np = np.arange(0, n_points, 1)
        index_np = np.hstack([index_np[:, None]])
        times_np = np.linspace(0, t_max, num=n_points)
        times_np = np.hstack([times_np[:, None]])


        for r in ray_in:
            F_ = np.max((r.detach().cpu().numpy())*(problem.target_space),axis = 1)
            y_target,sol_target = problem.target_space[F_.argmin(), :],problem.sols[F_.argmin(), :,:]
            func1 = TestODEF(Tensor([[-0.1, -0.5], [0.5, -0.1]]), Tensor([[0.2, 1.], [-1, 0.2]]), Tensor([[-1., 0.]]),s,r.detach().cpu().numpy())
            ode_true = NeuralODE(func1, solver_name, **solver_kwargs)
            r = r.unsqueeze(0)
            output = hnet(r)
            x0.append(output)
            times1 = torch.from_numpy(times_np[:, :, None]).to(output)
            obs = ode_true(output, times1, return_whole_sequence=True).detach()
            x_target.append(obs)

        return torch.stack(ray_in,dim=0),torch.stack(x0,dim=0),torch.stack(x_target,dim=0),times1
    best_epoch = 0
    for e in tqdm(range(200)):
        ode_trained.train()
        optimizer.zero_grad()
        ray_in,x0,x_target,times1 = create_dt(hnet,8,ray_train)
        obs_train = ode_trained(x0,times1,return_whole_sequence=True)
        obs_train = obs_train.permute(1,0,2,3)
        loss1 = [l1(obs_train[:,-1,:,:]),l2(obs_train[:,-1,:,:])]
        loss1 = torch.stack(loss1)
        loss2 = [l1(x_target[:,-1,:,:]),l2(x_target[:,-1,:,:])]
        loss2 = torch.stack(loss2)
        loss = torch.mean(torch.abs(obs_train - x_target))

        loss.backward()
        optimizer.step()
        print(f"Epoch: {e}, Loss: {loss}")
        t1 = time.time()
        hnet.eval()
        ode_trained.eval()
        tmp = []
        pf_pred = []
        pf_true = []

        for r in test_rays:
            r = r/(r.sum())
            ray_t = torch.tensor(r.tolist()).unsqueeze(0)
            F_ = np.max((ray_t.detach().cpu().numpy())*(problem.target_space),axis = 1)
            y_test,sol_test= problem.target_space[F_.argmin(), :],problem.sols[F_.argmin(), :,:]
            pf_true.append(y_test)
            x0_test = hnet(ray_t)

            z_test = ode_trained(x0_test.unsqueeze(0),times1, return_whole_sequence=True)
            obj1 = [l1(z_test[-1]).item(),l2(z_test[-1]).item()]
            pf_pred.append(obj1)
            obj1 = np.array(obj1)


            err = np.sum(np.abs(obj1-y_test))
            tmp.append(err)
        t2 = time.time()
        err_e = np.mean(np.array(tmp))

        if err_e < best_err:
            best_err = err_e
            best_epoch = e
            epochs_no_improve = 0
            print("ERR_test: ",best_err)
            torch.save(ode_trained.state_dict(), './best_model_nnode.pth')
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs with no improvement.")
            break
        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        print("Epoch %d: SGD lr %.4f -> %.4f" % (e, before_lr, after_lr))
    print("Neural ODE training complete.")
    return ode_trained, times1, best_epoch
