import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

class HeatPINN(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
    )
  def forward(self,x,t):
    inp=torch.cat((x,t),dim=1)
    return self.net(inp)


def pde_loss(f,x,t):
  u = f(x,t)

  du_dx=torch.autograd.grad(
    u,x,
    grad_outputs=torch.ones_like(u),
    create_graph=True
  )[0]

  d2u_dx2=torch.autograd.grad(
    du_dx,x,
    grad_outputs=torch.ones_like(du_dx),
    create_graph=True
  )[0]

  du_dt=torch.autograd.grad(
    u,t,
    grad_outputs=torch.ones_like(u),
    create_graph=True
  )[0]

  alpha=0.01
  heat_loss=torch.mean((du_dt-alpha*d2u_dx2)**2)
  return heat_loss

def bc_loss(f):
  x_bc_1=torch.zeros([100,1])
  x_bc_0=torch.ones([100,1])

  t_bc=torch.linspace(0,1,100).unsqueeze(1)
  
  u_bc_0=f(x_bc_0,t_bc)
  u_bc_1=f(x_bc_1,t_bc)

  bc_loss=torch.mean((u_bc_0)**2)+torch.mean((u_bc_1)**2)
  return bc_loss

def ic_loss(f,x):
  t_ic=torch.zeros([100,1])
  u_pre=f(x,t_ic)
  u_true=torch.sin(np.pi*x)
  u_ic=u_pre-u_true
  return torch.mean(u_ic**2)

def train_model(f,x,t,n_epoch=10000):
  optimizer=Adam(f.parameters(),lr=1e-3)
  scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

  losses=[]

  for i in range(n_epoch):
    optimizer.zero_grad()

    loss=pde_loss(f,x,t)+bc_loss(f)+ic_loss(f,x)
    loss.backward()

    optimizer.step()
    scheduler.step()

    losses.append(loss.item())

    if (i+1)%500==0:
      print(f'epoch {i+1:5d} and loss {loss.item():.8f}')
  return f , np.array(losses)



f = HeatPINN()
x = torch.randn(100,1).requires_grad_(True)
t = torch.randn(100,1).requires_grad_(True)

f, losses = train_model(f,x,t)

x_plot = torch.linspace(0, 1, 200).unsqueeze(1)
t_slices = [0.0, 0.25, 0.5, 0.75, 1.0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for t_val in t_slices:
    t_plot = torch.full((200, 1), t_val)

    with torch.no_grad():
        u_pred = f(x_plot, t_plot)

    u_exact = np.sin(np.pi * x_plot.numpy()) * np.exp(-0.01 * np.pi**2 * t_val)


    ax1.plot(x_plot.numpy(), u_exact, 'b-',  alpha=0.7, linewidth=2)
    ax1.plot(x_plot.numpy(), u_pred,  'r--', alpha=0.7, linewidth=2)



ax1.set_xlabel('x')
ax1.set_ylabel('u(x, t)')
ax1.set_title('Heat Equation — Exact (blue) vs PINN (red)\nt = 0, 0.25, 0.5, 0.75, 1.0')
ax1.grid(True)

ax2.semilogy(losses)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss (log scale)')
ax2.set_title('Training Loss')
ax2.grid(True)

plt.tight_layout()
plt.show()

t_test = torch.full((200, 1), 0.5)
with torch.no_grad():
    u_pred = f(x_plot, t_test).numpy()
u_exact = np.sin(np.pi * x_plot.numpy()) * np.exp(-0.01 * np.pi**2 * 0.5)
l2 = np.sqrt(np.mean((u_pred - u_exact)**2))
print(f'L2 Error at t=0.5: {l2:.6f}')
