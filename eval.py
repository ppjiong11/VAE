import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(5760 * 16, 512)
        self.fc2_mean = nn.Linear(512, 256)
        self.fc2_logvar = nn.Linear(512, 256)

        # Decoder
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 5760)
        self.conv4 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

        # Loss function
        self.L2_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCELoss(reduction='sum')

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 5760 * 16)
        x = F.relu(self.fc1(x))
        mean = self.fc2_mean(x)
        logvar = self.fc2_logvar(x)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def decode(self, z):
        x = F.relu(self.fc3(z))
        x = F.relu(self.fc4(x))
        x = x.view(-1, 1, 5760)
        #x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = torch.sigmoid(self.conv6(x))
        #x = torch.squeeze(x, dim=1) # 删除x中长度为1的维度，dim=1则指定了，如果dim1的维度长度为1，则删除此维度。
        return x

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar

    def loss_function(self, x, x_hat, mean, logvar):
        L2 = self.L2_loss(x_hat, x)
        KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return L2 + KLD
    

model = VAE()
model.load_state_dict(torch.load('model_epoch_2100.pt'))

def generate(model, num_samples):
    with torch.no_grad():
        data = np.load('cirs.npy')
        random_index = random.randint(0, len(data) - 1)
        random_data = np.take(data, random_index, axis=0)
        z = random_data
        z = z.reshape(-1, 1, z.shape[0])
        z = torch.tensor(z, dtype=torch.float32)
        z = z.to(next(model.parameters()).device)
        mean, logvar = model.encode(z)
        z = model.reparameterize(mean, logvar)
        x_hat = model.decode(z)
        x_hat = x_hat.cpu().numpy().reshape(num_samples, -1)
        return random_data, x_hat
    
def eval(model, num_samples): #从decoder中生成数据
     with torch.no_grad():
        z = torch.randn(num_samples, 256)
        z = z.to(next(model.parameters()).device)
        x_hat = model.decode(z)
        x_hat = x_hat.cpu().numpy().reshape(num_samples, -1)
        return x_hat

# 调用 generate() 函数，并传递之前加载的模型作为参数
random_data, new_data = generate(model, num_samples=1)
g_data = eval(model, num_samples=1)

fig, axs = plt.subplots(3, 1, figsize=(12, 10))
plt.subplots_adjust(hspace=0.5)
axs[0].plot(random_data)
axs[0].set_title('Original Data')
axs[0].set_ylim([0, 1])
axs[1].plot(new_data[0])
axs[1].set_title('Generated Data')
axs[1].set_ylim([0, 1])
axs[2].plot(g_data[0])
axs[2].set_title('Eval Generated Data')
axs[2].set_ylim([0, 1])
plt.show()