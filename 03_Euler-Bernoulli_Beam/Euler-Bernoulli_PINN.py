import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

class Euler_Bernoulli(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
    )
  def forward(self,x):
    return self.net(x)


def pde_loss(f,x):
  w = f(x)

  dw_dx=torch.autograd.grad(
    w,x,
    grad_outputs=torch.ones_like(w),
    create_graph=True
  )[0]

  d2w_dx2=torch.autograd.grad(
    dw_dx,x,
    grad_outputs=torch.ones_like(dw_dx),
    create_graph=True
  )[0]

  d3w_dx3=torch.autograd.grad(
    d2w_dx2,x,
    grad_outputs=torch.ones_like(d2w_dx2),
    create_graph=True
  )[0]

  d4w_dx4=torch.autograd.grad(
    d3w_dx3,x,
    grad_outputs=torch.ones_like(d3w_dx3),
    create_graph=True
  )[0]

  loss= torch.mean(((E*I*d4w_dx4)-q)**2)
  return loss

def bc_loss(f):
  x_bc_0=torch.zeros([1,1], requires_grad=True)
  x_bc_1=torch.ones([1,1], requires_grad=True)

  w_0=f(x_bc_0)
  w_1=f(x_bc_1)

  dw_0 =torch.autograd.grad(
    w_0,x_bc_0,
    grad_outputs=torch.ones_like(w_0),
    create_graph=True
  )[0]

  M_0=torch.autograd.grad(
    dw_0,x_bc_0,
    grad_outputs=torch.ones_like(dw_0),
    create_graph=True
  )[0]

  dw_1 =torch.autograd.grad(
    w_1,x_bc_1,
    grad_outputs=torch.ones_like(w_1),
    create_graph=True
  )[0]

  M_1=torch.autograd.grad(
    dw_1,x_bc_1,
    grad_outputs=torch.ones_like(dw_1),
    create_graph=True
  )[0]
  bc_loss=torch.mean(w_0**2)+torch.mean(w_1**2)+torch.mean(M_0**2)+torch.mean(M_1**2)
  return bc_loss



def train_model(f,x,n_epoch=10000):
  optimizer=Adam(f.parameters(),lr=1e-3)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epoch, eta_min=1e-5)
  
  losses=[]

  for i in range(n_epoch):
    optimizer.zero_grad()

    loss=pde_loss(f,x)+bc_loss(f)
    loss.backward()

    optimizer.step()
    scheduler.step()

    losses.append(loss.item())

    if (i+1)%500==0:
      print(f'epoch {i+1:5d} and loss {loss.item():.10f}')
  return f , np.array(losses)



E=1.0
I=1.0
L=1.0
q=1.0
f = Euler_Bernoulli()
x = torch.linspace(0, L, 100).view(-1, 1).requires_grad_(True)


f, losses = train_model(f,x)

x_plot = torch.linspace(0, 1, 200).unsqueeze(1)
w_exact = (q/(24*E*I))*((x_plot**4)-(2*L*(x_plot**3))+((L**3)*x_plot)).numpy()

with torch.no_grad():
    w_pred = f(x_plot).numpy()
l2 = np.sqrt(np.mean((w_pred - w_exact)**2))
print(f'L2 Error : {l2:.8f}')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(x_plot.numpy(), w_exact, 'b-',  alpha=0.7, linewidth=2)
ax1.plot(x_plot.numpy(), w_pred,  'r--', alpha=0.7, linewidth=2)
ax1.set_xlabel('x')
ax1.set_ylabel('w(x)')
ax1.set_title('Euler-Bernoulli Beam — Exact (blue) vs PINN (red)')
ax1.grid(True)

ax2.semilogy(losses)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss (log scale)')
ax2.set_title('Training Loss')
ax2.grid(True)

plt.tight_layout()
plt.show()

