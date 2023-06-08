import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
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
        self.L1_loss = nn.L1Loss(reduction='sum')
        self.bce_loss = nn.BCELoss(reduction='sum')
        self.L2_loss = nn.MSELoss(reduction='sum')
        self.NLL_loss = nn.NLLLoss(reduction='sum')

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.000001)

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
        #BCE = nn.functional.mse_loss(x_hat, x, reduction='sum')
        #NLL = self.NLL_loss(x_hat,x)
        L2 = self.L2_loss(x_hat, x)
        KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return 2*L2 + KLD

    def train_model(self, data, epochs):
        for epoch in range(epochs):
            # Shuffle the data
            #np.random.shuffle(data)

            # Split the data into batches
            random_indices = np.random.choice(len(data), size=32, replace=False)
            batches = np.take(data, random_indices, axis=0)

            #num_batches = len(data) // batch_size #计算数据集可以分成多少个批次
            #batches = np.array_split(data[:num_batches * batch_size], num_batches) #将数据集分成指定数量的批次
            # Train on each batch
            total_loss = 0
            for batch in batches:
                batch = batch.reshape(-1, 1, batch.shape[0])
                batch = torch.tensor(batch, dtype=torch.float32)

                    # Forward pass
                x_hat, mean, logvar = self(batch)

                    # Compute the loss
                loss = self.loss_function(batch, x_hat, mean, logvar)
                total_loss += loss.item()

                    # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Print the average loss for this epoch
            print(f"Epoch {epoch + 1}: average loss = {total_loss / len(data)}")
            if epoch % 300 == 0:
                save_model(model, f"model_epoch_{epoch}.pt")

    def generate(self, num_samples):#随机选择一条数据传入VAE，生成对应的数据
        with torch.no_grad():
            data = np.load('cirs.npy')
            random_index = random.randint(0, len(data) - 1)
            random_data = np.take(data, random_index, axis=0)
            z = random_data
            z = z.reshape(-1, 1, z.shape[0])
            z = torch.tensor(z, dtype=torch.float32)
            z = z.to(next(self.parameters()).device)
            mean, logvar=self.encode(z)
            z = self.reparameterize(mean, logvar)
            x_hat = self.decode(z)
            x_hat = x_hat.cpu().numpy().reshape(num_samples, -1)
            return random_data , x_hat
        
    def eval(self, num_samples): #从decoder中生成数据
        with torch.no_grad():
            z = torch.randn(num_samples, 256)
            z = z.to(next(self.parameters()).device)
            x_hat = self.decode(z)
            x_hat = x_hat.cpu().numpy().reshape(num_samples, -1)
            return x_hat
        
def save_model(model, filepath):
    """
    保存模型参数
    
    Args:
    - model: 模型
    - filepath: 模型参数保存路径
    """
    torch.save(model.state_dict(), filepath)

# Load the data
data = np.load('cirs.npy')
#random_index = random.randint(0, len(data) - 1)
#random_data = np.take(data, random_index, axis=0)
# Create a VAE model
model = VAE()

# Train the model
model.train_model(data, epochs=3001)
