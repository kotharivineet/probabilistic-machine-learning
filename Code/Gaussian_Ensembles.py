import matplotlib.pyplot as plt
from torch import nn, optim
import numpy as np
import torch

############## Parameters ##############

N = 1000
num_epoch = 50
lr = 0.003
ensemble_size = 10

############## Input Data ##############

Y = np.sin(np.linspace(0,10,N))
# Y = np.linspace(0,0,N)
# Y[200:800] = Y_sin[200:800]
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

ensemble = []
for i in range(ensemble_size):
    model = GMLPE()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.GaussianNLLLoss()
    ensemble.append((model, optimizer, loss_fn))

############## Training ###############

for epoch in range(num_epoch):
    for model, optimizer, criterion in ensemble:
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output[:,0], Y, output[:,1])
        loss.backward()
        optimizer.step()

############### Testing ###############

with torch.no_grad():
    y_pred = torch.stack([model(X) for model,_,_ in ensemble])

######### Managing Variables ##########

mean = torch.mean(y_pred[:,:,0], axis=0)
epistemic_std = torch.sqrt(torch.var(y_pred[:,:,0], axis=0))
aleatoric_std = torch.sqrt(torch.mean(y_pred[:,:,1], axis=0))
total_var = epistemic_std + aleatoric_std

################ Plots ################

plt.plot(mean, label='Prediction')
plt.fill_between([i for i in range(0,1000)], mean+2*total_var, mean-2*total_var, alpha=0.3, label='Epistemic Uncertainty',linewidth=0.0)
plt.fill_between([i for i in range(0,1000)], mean+2*aleatoric_std, mean-2*aleatoric_std, alpha=0.3, label='Aleatoric Uncertainty',linewidth=0.0)
plt.plot(Y, label='Ground Truth')
plt.scatter([i for i in range(200,800)],X_train, marker='.', label='Training Data Points')

# plt.ylim([-1,1])
plt.legend()
plt.show()