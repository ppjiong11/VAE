import torch# 导入PyTorch库
import torch.nn as nn# 导入神经网络模块
import torch.optim as optim# 导入优化器模块
from torch.autograd import Variable # 导入自动求导模块
from torchvision import datasets, transforms# 导入数据集和数据变换模块
import numpy as np # 导入NumPy库
from torch.utils.data import TensorDataset, DataLoader# 导入数据集和数据加载器
from load_data import data_npy

# 定义VAE模型
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        # 定义 VAE 的编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )
        # 定义 VAE 的解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        # 编码器的前向传递
        h = self.encoder(x)
        # 按照潜在变量的维度拆分输出
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # 重新参数化技巧，用于确保潜在向量的采样过程是可导的
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # 解码器的前向传递
        return self.decoder(z)

    def forward(self, x):
        # VAE 的前向传递
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# 定义VAE损失函数
def loss_function(recon_x, x, mu, logvar):
    # 计算重建误差
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum') #BCE = nn.MSELoss(reduction='sum')(recon_x, x)
    # 计算 KL 散度
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # 返回总损失
    return BCE + KLD

# 训练VAE模型
def train(model, train_loader, optimizer, epoch):
    # 进入训练模式
    model.train()
    train_loss = 0
    for batch_idx, (data,) in enumerate(train_loader):
        data = data.to(device)# 将数据移动到指定设备上
        optimizer.zero_grad()# 优化器梯度清零
        recon_batch, mu, logvar = model(data)# 前向传递
        loss = loss_function(recon_batch, data, mu, logvar)# 计算损失
        loss.backward()# 反向传播
        train_loss += loss.item()# 累计损失
        optimizer.step()# 权重更新
        if batch_idx % 10 == 0: #判断当前批次是否是 100 的倍数，如果当前批次是 100 的倍数，那么就会执行 print() 函数输出当前 epoch 中已经处理的数据量和进度百分比。
              print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')#每训练100个batch就打印一次训练情况 [当前epoch] [已处理样本数/总样本数 (已完成进度)] Loss: [当前batch的平均损失]

# 生成新的数据文件
def generate_data(model, num_samples):
    with torch.no_grad(): # 关闭梯度跟踪
        z = torch.randn(num_samples, latent_dim).to(device)# 从标准正态分布中采样潜在向量
        samples = model.decode(z).cpu().numpy()# 生成新的数据样本
    return samples

# 准备数据
data = data_npy      # 加载时间序列数据，形状为[num_samples, input_dim]
input_dim = data.shape[1]
hidden_dim = 64
latent_dim = 16
batch_size = 64 #每个批次处理64个数据，一共有3963个数据，所以batch_idx为61.92，% 10为6，所以一个epoch会print6次。
epochs = 30

# 将数据转换为TensorDataset
dataset = TensorDataset(torch.from_numpy(data).float())

# 创建数据加载器
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 定义模型、优化器和设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(input_dim, hidden_dim, latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 训练模型
for epoch in range(1, epochs + 1):
    train(model, train_loader, optimizer, epoch)

model.eval()
num_samples = 1000 # 生成1000个新数据样本
generated_data = generate_data(model, num_samples)

# 保存生成的数据
np.save('generated_data.npy', generated_data)