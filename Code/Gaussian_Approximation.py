import matplotlib.pyplot as plt
from torch import nn, optim
import numpy as np
import torch

############## Parameters ##############

N = 1000
num_epoch = 100
lr = 0.01

############## Input Data ##############

Y = np.sin(np.linspace(0,5,N))
# Y = np.linspace(0,0,N)
X = torch.tensor((np.ones(N)*Y + np.random.normal(0,0.1,N)),dtype=torch.float32)
Y = torch.tensor(Y,dtype=torch.float32)

X = torch.reshape(X,[len(X),1])
# Y = torch.reshape(Y,[len(Y),1])

############## Data Split ##############

X_train = X[200:800]
Y_train = Y[200:800]

X_train = X_train.type(torch.float32)
Y_train = Y_train.type(torch.float32)

################ Model #################
class GMLPE(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(1, 16)
        self.mid = nn.Linear(16,16)
        self.layer_end = nn.Linear(16,2)
        self.Softplus = nn.Softplus()

    def forward(self, x):
        x = self.lin(x)
        x = self.mid(x)
        x = self.layer_end(x)
        x[:,1] = self.Softplus(x[:,1].clone())
        return x

################ Setup #################

model = GMLPE()

loss_fn = nn.GaussianNLLLoss()

optimizer = optim.Adam(model.parameters(), lr)

############## Training ###############

for i in range(num_epoch):
    y_pred = model(X_train)
    loss = loss_fn(y_pred[:,0], Y_train, y_pred[:,1])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

############### Testing ###############

with torch.no_grad():
    y_pred = model(X)

######### Managing Variables ##########

mean = y_pred[:,0].detach().numpy()
std = np.sqrt(y_pred[:,1].detach().numpy())

################ Plots ################

plt.plot(mean, label='Prediction')
plt.fill_between([i for i in range(0,1000)], mean+2*std, mean-2*std, alpha=0.5, label='Aleatoric Uncertainty',linewidth=0.0)
plt.plot(Y, label='Ground Truth')
plt.scatter([i for i in range(200,800)],X_train, marker='.', label='Training Data Points')

# plt.ylim([-1,1])
plt.legend()
plt.show()