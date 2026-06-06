import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

class my_nn(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
    )
  def forward(self,t):
    return self.net(t)


def ode_loss(f,x):
  u = f(x)
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

  g=-torch.pi**2 * torch.sin(torch.pi * x)
  ode_loss=torch.mean((d2u_dx2-g)**2)
  return ode_loss

def bc_loss(f):
  x_bc=torch.tensor([0.0,1.0]).view(-1,1)
  u_bc=f(x_bc)
  return torch.mean((u_bc)**2)


def train_model(f,x,n_epoch=10000):
  optimizer=Adam(f.parameters(),lr=1e-3)
  scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

  losses=[]

  for i in range(n_epoch):
    optimizer.zero_grad()

    loss=ode_loss(f,x)+bc_loss(f)
    loss.backward()

    optimizer.step()
    scheduler.step()

    losses.append(loss.item())

    if (i+1)%500==0:
      print(f'epoch {i+1:5d} and loss {loss.item():.8f}')
  return f , np.array(losses)



f = my_nn()
x = torch.linspace(0, 1, 100).unsqueeze(1).requires_grad_(True)
f, losses = train_model(f, x)


x_test = torch.linspace(0, 1, 200).unsqueeze(1)
with torch.no_grad():
    u_pred = f(x_test).numpy()

u_exact = np.sin(np.pi * x_test.numpy())

with torch.no_grad():
    u_pred_test = f(x_test).numpy()
l2_error = np.sqrt(np.mean((u_pred_test - u_exact)**2))
print(f'\nL2 Error: {l2_error:.6f}')


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))


ax1.plot(x_test.numpy(), u_exact, 'b-', label='Exact', linewidth=2)
ax1.plot(x_test.numpy(), u_pred, 'r--', label='PINN', linewidth=2)
ax1.set_xlabel('x')
ax1.set_ylabel('u(x)')
ax1.set_title('PINN vs Exact Solution')
ax1.legend()
ax1.grid(True)

ax2.semilogy(losses)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss (log scale)')
ax2.set_title('Training Loss')
ax2.grid(True)

plt.tight_layout()
plt.show()


