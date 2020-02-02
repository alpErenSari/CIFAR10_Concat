import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from models import NetWide
import argparse


parser = argparse.ArgumentParser(description='Classification with two concatnated images on CIFAR10 dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--epochs', default=4, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')

args = parser.parse_args()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Prepare the dataset

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=args.workers)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.workers)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')




net_wide = NetWide().to(device)

criterion = nn.BCELoss()
optimizer = optim.SGD(net_wide.parameters(), lr=args.lr, momentum=args.momentum)

for epoch in range(args.epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        labels_one = F.one_hot(labels, num_classes=10).float()

        index = torch.randperm(4)
        inputs_2 = inputs[index, :,:,:]
        labels_2 = labels[index]
        labels_one_2 = F.one_hot(labels_2, num_classes=10).float()
        
        inputs_wide = torch.cat((inputs, inputs_2), dim=-1)
        labels_wide = 0.5*labels_one + 0.5*labels_one_2

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net_wide(inputs_wide)
        loss = criterion(outputs, labels_wide)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# save the trained weights
PATH = './cifar_net.pth'
torch.save(net_wide.state_dict(), PATH)
print('Network weights saved')

# Test the network
# Load the trained network
net_wide = NetWide()
net_wide.load_state_dict(torch.load(PATH))

# calculate the accuracy from the first image over the test data
net_wide = net_wide.to(device)
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        index = torch.randperm(4)
        images_2 = images[index, :,:,:]
        labels_2 = labels[index]
        images_wide = torch.cat((images,images_2), dim=-1)
        outputs = net_wide(images_wide)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))



