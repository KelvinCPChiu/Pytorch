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
        #x = torch.softmax(x, dim=0)
        return x


def train_net(batch_size):

    print('Loading the Dataset')

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    print('Building Model')

    net = MobileNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001) #momentum=0.9)

    print('Training the Model')
    for epoch in range(10):  # loop over the dataset multiple times

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
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    torch.save({'model': net.state_dict()}, './model.pth')


def test_net(batch_size):

    net = MobileNet()

    net.load_state_dict(torch.load('./model.pth')['model'])
    net.to(device)

    class_correct = list([0]*10)
    class_total = list([0]*10)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    with torch.no_grad():
        for data in testloader:
            images, labels = data

            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            #print(c.shape)
            for i in range(batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':
    train_net(50)
    test_net(500)
