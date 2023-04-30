import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from operator import itemgetter


ratings_title = ['UserID','MovieID', 'Rating', 'timestamps']
ratings = pd.read_csv('./dataset/u.data', sep='\t', header=None, names=ratings_title, engine = 'python')

ratings.sample(frac=1.0)
train_set, test_set = train_test_split(ratings,test_size = 0.2)
dev_set, test_set = train_test_split(test_set,test_size = 0.5)
train_set = np.array(train_set, dtype = 'int')
dev_set = np.array(dev_set, dtype = 'int')
test_set = np.array(test_set, dtype = 'int')

nb_users = int(max(max(train_set[:, 0]), max(dev_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(train_set[:, 1]), max(dev_set[:, 1]), max(test_set[:, 1])))


# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    # 匹配相对应的 local data里面 user与movie相匹配，若user没有评价过该movie，then mark ‘0’
    for id_users in range(1, nb_users + 1):
        # create 第一层list，id_movies为某user看过的movie的id
        id_movies = data[:, 1][data[:, 0] == id_users]
        # id_ratings为某user评价过某movie的rate
        id_ratings = data[:, 2][data[:, 0] == id_users]
        # 首先创建全部为0的list，再将user 评价movie的rating分数替换0，那么就能mark user没看过的movie为0
        ratings = np.zeros(nb_movies)
        # 由于movieID由1开始，而python由0开始，因此要rating匹配python则-1
        ratings[id_movies - 1] = id_ratings
        # 将以上创建的list合并到一个list，以被torch提取
        new_data.append(list(ratings))
    return new_data


train_set = convert(train_set)
dev_set = convert(dev_set)
test_set = convert(test_set)

train_set = torch.FloatTensor(train_set)
train_set = train_set[torch.sum(train_set != 0, axis=1) != 0]  # 删除评分全为0的行
dev_set = torch.FloatTensor(dev_set)
dev_set = dev_set[torch.sum(dev_set != 0, axis=1) != 0]  # 删除评分全为0的行
test_set = torch.FloatTensor(test_set)
test_set = test_set[torch.sum(test_set != 0, axis=1) != 0]  # 删除评分全为0的行

train_dataset = TensorDataset(train_set)
dev_dataset = TensorDataset(dev_set)
test_dataset = TensorDataset(test_set)

train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
dev_loader = DataLoader(dataset=dev_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)


class AE(nn.Module):
  """
  Auto Encoder(AE)最初用于学习数据的表示(编码)。其可被分解为两部分:

  encoder：减少了数据的维度大小;
  decoder：它将编码转换回其原始形式。由于存在降维，神经网络需要学习输入(潜在空间)的低维表示，以便能够重构输入。

  它们可以用来预测新的推荐。为了做到这一点，输入和输出都是点击向量(通常AE的输入和输出是相同的)，我们将在输入层之后使用大的dropout。
  这意味着模型将不得不重构点击向量，因为输入中的某个元素将会丢失，因此要学会预测给定的点击向量的推荐值。
  """
  def __init__(self, nb_movies, device="cuda:0"):
    super(AE, self).__init__()
    self.nb_movies = nb_movies
    self.encoder = nn.Sequential(
        nn.Linear(self.nb_movies, 512),
        nn.Sigmoid(),
        nn.Dropout(0.9), # 这里需要一个大的dropout
        nn.Linear(512, 80),
        nn.Sigmoid(),
        nn.Linear(80, 32),
        nn.Sigmoid()
        )
    self.decoder = nn.Sequential(
        nn.Linear(32, 80),
        nn.Sigmoid(),
        nn.Linear(80, 512),
        nn.Sigmoid(),
        nn.Linear(512, self.nb_movies)
        )

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

def loss_func(recon_x, x):
  """
  对于一个用户而言，他只对部分电影进行了打分，因此在计算MSE时，只考虑他打过分的电影，而忽略他没打过分的电影
  这里MSE的计算原理是，例如2个用户对5个电影的打分数据为：[[1, 2, 0, 3, 4], [2, 3, 0, 0, 1]]
  经过AE重构后的打分数据为：[[1.1, 2.3, 0, 3.3, 4.7], [2.1, 3.2, 0, 0, 1.2]]
  则先计算两打分数据的2范数的平方，再除以每个用户打过分的电影数，得到每个用户的MSE，再用torch.mean求平均得到每个batch的MSE
  """
  MSE = torch.mean(torch.norm((x - recon_x), p=2, dim=1, keepdim=False)**2/torch.sum(recon_x!=0,axis=1))
  return MSE



def train(train_loader, dev_loader=None, is_validate=True, device="cuda:0"):
  ae.train()
  total_loss = 0
  for _, data in enumerate(train_loader, 0):
    data = Variable(data[0]).to(device)
    target = data.clone()
    optimizer.zero_grad()
    recon_x = ae.forward(data)
    # 在优化过程中，我们只想考虑用户打过分的电影，
    # 虽然我们之前将用户未打分的电影的评分设为0，但是我们还需要将模型预测的用户未打分的电影的评分也设为0，
    # 这样才不会累加到loss里导致影响了权重更新
    recon_x[target == 0] = 0
    loss = loss_func(recon_x, data)
    loss.backward()
    total_loss += loss.item()
    optimizer.step()
    epoch_train_loss = total_loss / len(train_loader)

  if (is_validate == True):
    ae.eval()
    total_loss = 0
    with torch.no_grad():
      for _, data in enumerate(dev_loader, 0):
        data = Variable(data[0]).to(device)
        target = data.clone()
        recon_x = ae.forward(data)
        recon_x[target == 0] = 0
        loss = loss_func(recon_x, data)
        total_loss += loss.item()
        epoch_dev_loss = total_loss / len(dev_loader)
    print('====> Epoch: {} Training Average loss: {:.4f}, Validating Average loss: {:.4f}'.format(epoch, epoch_train_loss, epoch_dev_loss))
    return epoch_train_loss, epoch_dev_loss

  else:
    print('====> Epoch: {} Training Average loss: {:.4f}'.format(epoch, epoch_train_loss))
    return epoch_train_loss

def test(test_loader, device="cuda:0"):
  ae.eval()
  total_loss = 0
  with torch.no_grad():
    for _, data in enumerate(test_loader, 0):
      data = Variable(data[0]).to(device)
      target = data.clone()
      recon_x = ae.forward(data)
      recon_x[target == 0] = 0
      loss = loss_func(recon_x, data)
      total_loss += loss.item()
  print('Average test loss: {:.4f}'.format(total_loss / len(test_loader)))

if __name__ == "__main__":
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  ae = AE(nb_movies = nb_movies).to(device)
  optimizer =  optim.Adam(ae.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
  EPOCH = 200
  epoches_train_loss = []
  epoches_dev_loss = []

  print("\n", 20 * "=", "Training Auto Encoder on device: {}".format(device), 20 * "=")
  for epoch in range(1, EPOCH + 1):
    epoch_train_loss, epoch_dev_loss = train(train_loader, dev_loader = dev_loader)
    epoches_train_loss.append(epoch_train_loss)
    epoches_dev_loss.append(epoch_dev_loss)
  print("\n", 20 * "=", "Testing Auto Encoder on device: {}".format(device), 20 * "=")
  test(test_loader)

  # save the training model
  torch.save(ae.state_dict(), 'AutoEncoder.pkl')