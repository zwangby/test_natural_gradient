#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: zhixiang time:18-7-3



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# transforms do transformation to dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import kfac

# Training settings
batch_size = 64

# MNIST
train_dataset = datasets.MNIST(root='./data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               # data_tf = transforms.Compose(
                               # [transforms.ToTensor(),
                               #  transforms.Normalize([0.5], [0.5])])
                               download=True)

test_dataset = datasets.MNIST(root='./data/',
                              train=False,
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # input 1 channel, output 10 channels, kernel 5*5
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        # fully connect
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        # in_size = 64
        in_size = x.size(0) # one batch
        # x: 64*16*12*12
        x = F.relu(self.mp(self.conv1(x)))
        # x: 64*32*4*4
        x = F.relu(self.mp(self.conv2(x)))
        # x: 64*512
        x = x.view(in_size, -1) # flatten the tensor
        # x: 64*10
        x = self.fc(x)
        return F.log_softmax(x)


model = Net()


optimizer = kfac.KFACOptimizer(model)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        l2_reg = torch.Tensor([0])
        for param in model.parameters():
            l2_reg += torch.norm(param)
        loss = F.cross_entropy(output, target)
        # loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 200 == 0:
             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                 epoch, batch_idx * len(data), len(train_loader.dataset),
                 100. * batch_idx / len(train_loader), loss.data[0]))


def test():
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        # test_loss += F.nll_loss(output, target, size_average=False).data[0]
        # test_loss += torch.nn.NLLLoss(output, target, size_average=False).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\n Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
          test_loss, correct, len(test_loader.dataset),
          100. * correct / len(test_loader.dataset)))


for epoch in range(1, 50):
    train(epoch)
    test()