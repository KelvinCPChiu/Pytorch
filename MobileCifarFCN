import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using {}'.format(device))


class depthwise_pointwise_module(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(depthwise_pointwise_module, self).__init__()
        self.conv_depth = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, groups=in_channels)
        self.BN_1 = nn.BatchNorm2d(num_features=in_channels)
        self.conv_point = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.BN_2 = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        x = self.conv_depth(x)
        x = self.BN_1(x)
        x = F.relu(x)
        x = self.conv_point(x)
        x = self.BN_2(x)
        x = F.relu(x)
        return x


class MobileNet(nn.Module):

    def __init__(self):
        super(MobileNet, self).__init__()
        self.conv_s1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv_dw1 = depthwise_pointwise_module(in_channels=32, out_channels=64)
        self.conv_dw2 = depthwise_pointwise_module(in_channels=64, out_channels=128)
        self.conv_s3 = nn.Conv2d(in_channels=128, out_channels=10, kernel_size=1)

    def forward(self, x):
        x = self.conv_s1(x)
        x = self.conv_dw1(x)
        x = self.conv_dw2(x)
        x = self.conv_s3(x)
        x = torch.sum(torch.sum(x, dim=2), dim=2)
        x = torch.softmax(x, dim=0)
        return x

print('Loading the Dataset')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=5,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=5,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print('Building Model')

net = MobileNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print('Training the Model')
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

torch.save({'model': net.state_dict()}, './model.pth')
