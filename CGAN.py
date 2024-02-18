"""
This is a distributed training script
分散型トレーニングスクリプトです

DATA SOURCE
データソース:
            https://github.com/liaopeiyuan/artbench

USAGE
用法:
            spark-submit CGAN.py \
            # custom source assignment, for example, カスタムリソース割り当て,たとえば、
            --master <your Spark standalone url あなたのSpark standalone url> \
            --total-executor-cores 4 \
            --executor-cores 1 \
            --executor-memory 2G \
            --driver-memory 4G \

COMPILED BY
コンパイラ:
            Chung Wong
            キカイチュウ

REFERRED TO
リファレンス:
            Andrew Gu's https://github.com/pytorch/examples/blob/1aa2eec9ac94102ac479cd88396b8aa3f2429092/distributed/ddp/example.py
"""

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from typing import Union, Literal, Optional
from pyspark.ml.torch.distributor import TorchDistributor
from pyspark.sql import SparkSession

"""
Parameters pre-setting
事前のパラメータ設定
"""
noise_len = 100#length of noise input
batch_size = 256#batch size
d_learning_rate, g_learning_rate = 0.0001, 0.0001#learning rates of discriminator and generator
epochs_n = 100#number of training epochs
shown_stride = 5#batch interval stride size to supervise sampling which must be less than (epochs_n / batch_size)

"""
Network
ニューラルネットワーク
"""
class D(nn.Module):
    """
    Discriminator of Convolutional Generative Adversarial Network
    コンボリューション生成対抗ネットワークの識別器
    """
    def __init__(self):
        """
        Initialization
        初期化
        """
        super(D, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size = 13, stride = 3, bias = False)
        self.conv2 = nn.Conv2d(6, 9, kernel_size = 7, stride = 3, bias = False)
        self.conv3 = nn.Conv2d(9, 12, kernel_size = 5, stride = 3, bias = False)
        self.conv4 = nn.Conv2d(12, 15, kernel_size = 5, stride = 1, bias = False)
        self.linear = nn.Linear(240, 1)
        self.norm1 = nn.BatchNorm2d(6)
        self.norm2 = nn.BatchNorm2d(9)
        self.norm3 = nn.BatchNorm2d(12)
        self.norm4 = nn.BatchNorm2d(15)
        self.activation = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input: torch.Tensor):#[batch_size, 3, 256, 256]
        """
        Forward propagation
        フォワード伝播
        """
        output = self.dropout(self.activation(self.norm1(self.conv1(input))))#[batch_size, 6, 82, 82],82=(256-13)/3+1
        output = self.dropout(self.activation(self.norm2(self.conv2(output))))#[batch_size, 9, 26, 26],26=(82-7)/3+1
        output = self.dropout(self.activation(self.norm3(self.conv3(output))))#[batch_size, 12, 8, 8],8=(26-5)/3+1
        output = self.dropout(self.activation(self.norm4(self.conv4(output))))#[batch_size, 15, 4, 4],4=(8-5)/1+1
        output = output.reshape(-1, 240)#[batch_size, 240],240=15X4X4
        output = self.activation(self.linear(output)).reshape(-1)#[batch_size]

        return output

    def init_model(self):
        """
        Model initialization
        モデル初期化
        """
        for name, layer in self._modules.items():
            if name in ('conv1', 'conv2', 'conv3'):
                nn.init.kaiming_normal_(layer.weight, mode = 'fan_in', nonlinearity = 'leaky_relu')
            elif name == 'conv4':
                nn.init.kaiming_normal_(layer.weight, mode = 'fan_out', nonlinearity = 'leaky_relu')
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1.)
                nn.init.constant_(layer.bias, 0.)
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std = 0.01)
                nn.init.normal_(layer.bias, std = 0.01)

    def loadparam(self, path: str):
        """
        Load pre-saved model parameters
        ロード前に保存したモデルパラメータ
        """
        self.load_state_dict(torch.load(path, map_location = torch.device('cpu')))

    def savemodel(self, path: str = './Discriminator.pth'):
        """
        Save the trained model
        訓練されたモデルを保存する
        """
        torch.save(self.state_dict(), path)

    def showparam(self):
        """
        Show model parameters
        モデルパラメータの表示
        """
        return [(name, param) for name, param in self.named_parameters()]

class G(nn.Module):
    """
    Generator of Convolutional Generative Adversarial Network
    コンボリューション生成対抗ネットワークの生成器
    """
    def __init__(self):
        """
        Initialization
        初期化
        """
        super(G, self).__init__()
        self.t_conv1 = nn.ConvTranspose2d(15, 12, kernel_size = 5, stride = 1, bias = False)
        self.t_conv2 = nn.ConvTranspose2d(12, 9, kernel_size = 5, stride = 3, bias = False)
        self.t_conv3 = nn.ConvTranspose2d(9, 6, kernel_size = 7, stride = 3, bias = False)
        self.t_conv4 = nn.ConvTranspose2d(6, 3, kernel_size = 13, stride = 3, bias = False)
        self.linear = nn.Linear(100, 240)
        self.activation = nn.LeakyReLU(0.2)
        self.norm1 = nn.BatchNorm2d(12)
        self.norm2 = nn.BatchNorm2d(9)
        self.norm3 = nn.BatchNorm2d(6)
        self.norm4 = nn.BatchNorm2d(3)
        self.tanh = nn.Tanh()

    def forward(self, input: torch.Tensor):#[batch_size, 100]
        """
        Forward propagation
        フォワード伝播
        """
        output = self.activation(self.linear(input))#[batch_size, 240]
        output = output.reshape(-1, 15, 4, 4)#[batch_size, 15, 4, 4]其中240=15X4X4
        output = self.activation(self.norm1(self.t_conv1(output)))#[batch_size, 15, 8, 8]
        output = self.activation(self.norm2(self.t_conv2(output)))#[batch_size, 9, 26, 26]
        output = self.activation(self.norm3(self.t_conv3(output)))#[batch_size, 6, 82, 82]
        output = self.activation(self.norm4(self.t_conv4(output)))#[batch_size, 3, 256, 256]
        output = self.tanh(output)#[batch_size, 3, 256, 256]

        return output

    def init_model(self):
        """
        Model initialization
        モデル初期化
        """
        for name, layer in self._modules.items():
            if name in ('t_conv1', 't_conv2', 't_conv3'):
                nn.init.kaiming_normal_(layer.weight, mode = 'fan_in', nonlinearity = 'leaky_relu')
            elif name == 't_conv4':
                nn.init.kaiming_normal_(layer.weight, mode = 'fan_out', nonlinearity = 'leaky_relu')
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1.)
                nn.init.constant_(layer.bias, 0.)
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std = 0.01)
                nn.init.normal_(layer.bias, std = 0.01)

    def loadparam(self, path: str):
        """
        Load pre-saved model parameters
        ロード前に保存したモデルパラメータ
        """
        self.load_state_dict(torch.load(path, map_location = torch.device('cpu')))

    def savemodel(self, path: str = './Generator.pth'):
        """
        Save the trained model
        訓練されたモデルを保存する
        """
        torch.save(self.state_dict(), path)

    def showparam(self):
        """
        Show model parameters
        モデルパラメータの表示
        """
        return [(name, param) for name, param in self.named_parameters()]

"""
Training assistance
訓練の補助
"""
class GAN_training_dominator():
    """
    GAN训练控制器
    GANトレーニングのコントローラ
    """
    def __init__(self, 
                 generator, 
                 discriminator, 
                 train_loader: torch.utils.data.dataloader.DataLoader, 
                 noise_len: int, 
                 device: Optional[Literal['cpu', 'cuda']] = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialization
        初期化
        """
        super(GAN_training_dominator, self).__init__()
        self.g_cost_log, self.d_cost_log = torch.tensor([]), torch.tensor([])#history of cost during training generator and discriminator
        self.generator, self.discriminator = generator, discriminator
        self.train_loader = train_loader
        self.noise_len = noise_len
        self.device = device
    '''-------------------------------------------------------------------------------------------'''
    def train(self, 
              g_optimizer, 
              d_optimizer, 
              epochs_n: Union[int, torch.Tensor] = 100, 
              shown_stride: Union[int, torch.Tensor] = 10, 
              cost_fn = nn.BCEWithLogitsLoss()):
        """
        Training
        訓練
        """
        self.generator.train()
        self.discriminator.train()
        for epoch in range(1, epochs_n + 1):
            g_train_losses, d_train_losses = torch.tensor(0.), torch.tensor(0.)#cost history during initializing training

            for x, _ in self.train_loader:
                x, noise = x.to(self.device), torch.randn(x.shape[0], self.noise_len, device = self.device)
                true_lbs, false_lbs = torch.ones(x.shape[0], device = self.device), torch.zeros(x.shape[0], device = self.device)
                #Stage 1: Use real samples to train discriminator
                pred1 = self.discriminator(x)
                d_cost1 = cost_fn(pred1, true_lbs)

                #Stage 2: Use Generated samples to train discriminator
                pred2 = self.discriminator(self.generator(noise).detach())
                d_cost2 = cost_fn(pred2, false_lbs)
                d_cost = d_cost1 + d_cost2
                d_optimizer.zero_grad()
                d_cost.backward()
                d_optimizer.step()

                #Stage 3: Use Generated samples to train generator
                pred3 = self.discriminator(self.generator(noise))
                g_cost = cost_fn(pred3, true_lbs)
                g_optimizer.zero_grad()
                g_cost.backward()
                g_optimizer.step()

                d_train_losses += d_cost.cpu().detach() * x.shape[0]#losses in each batch
                g_train_losses += g_cost.cpu().detach() * noise.shape[0]#losses in each batch

            d_train_cost, g_train_cost = d_train_losses / len(self.train_loader.dataset), g_train_losses / len(self.train_loader.dataset)#losses in each epochs
            self.d_cost_log, self.g_cost_log = torch.cat([self.d_cost_log, d_train_cost.reshape(-1)]), torch.cat([self.g_cost_log, g_train_cost.reshape(-1)])

            if epoch % shown_stride == 0 or epoch == 1:
                print(f'Epoch {epoch}, training cost of discriminator:{d_train_cost.item():.4f}, training cost of generator:{g_train_cost.item():.4f}')

"""
Import data and preprocess data
データのインポートと前処理データ
"""
norm_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])#normalize images
trainset = ImageFolder('./artbench-10-imagefolder-split/train', transform = norm_trans)

def train(epochs_n: int = 100, g_learning_rate: float = 0.0001, d_learning_rate: float = 0.0001, shown_stride = 5):
    """
    Distributed training
    分散トレーニング
    """
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader, DistributedSampler
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    dist.init_process_group('gloo')#distributed environment initialization

    ori_disc, ori_gen = D(), G()#instantiate model
    ori_disc.init_model()#initialize model
    ori_gen.init_model()#initialize model
    disc, gen = DDP(ori_disc), DDP(ori_gen)#parallelize model

    distributed_sampler = DistributedSampler(trainset)#distributed slices of samples
    train_loader = DataLoader(dataset = trainset, batch_size = batch_size, sampler = distributed_sampler, shuffle = False)

    g_optimizer = optim.Adam(gen.parameters(), lr = g_learning_rate)
    d_optimizer = optim.Adam(disc.parameters(), lr = d_learning_rate)

    dominator = GAN_training_dominator(gen, disc, train_loader, noise_len)
    dominator.train(g_optimizer = g_optimizer, d_optimizer = d_optimizer, epochs_n = epochs_n, shown_stride = shown_stride)

    dist.destroy_process_group()#cleaning up distributed environments
    disc.module.savemodel()#saving network weights parameters and it should be the default options to save the main process(rank=0)
    gen.module.savemodel()#saving network weights parameters and it should be the default options to save the main process(rank=0)
    torch.save(dominator.d_cost_log, './Discriminator_log.pth')
    torch.save(dominator.g_cost_log, './Generator_log.pth')

"""
distributed execution
分散計算の実行
"""
if __name__ == '__name__':
    spark = SparkSession.builder.appName('Artbench').getOrCreate()

    distributor = TorchDistributor(num_processes = 2, #number of processes
                                local_mode = False, #Using local mode or computing clusters
                                use_gpu = False)
    print('-'*20 + 'start training' + '-'*20)
    distributor.run(train, epochs_n, g_learning_rate, d_learning_rate, shown_stride)#Users can tune the training parameters according to the situation
    print('-'*20 + 'training done' + '-'*20)

    spark.stop()