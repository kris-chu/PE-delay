import torch
import torch.nn as nn
import torchvision
import functionsal_1 as f
import torch.utils.data.dataloader as Data
from torch.autograd import Variable
from Conv2D import Conv2d
from Linear import Linear
from cifar import CIFAR10
from container import Sequential, ReLU, MaxPool2d


batch_size = 1

# train_data = CIFAR10('./cifar', train=True, transform=torchvision.transforms.ToTensor())
test_data = CIFAR10(
    './cifar', train=False, transform=torchvision.transforms.ToTensor()
)
print(type(test_data))

# train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class vgg_cifar10(nn.Module):
    '''
    VGG model for CIFAR-10
    '''
    def __init__(self):
        super(vgg_cifar10, self).__init__()

        self.features = Sequential(
            Sequential(
                Conv2d(3, 64, 3, padding=3),
                ReLU(inplace=True),
                MaxPool2d(kernel_size=2, stride=2)
            ),
            Sequential(
                Conv2d(64, 128, 3, padding=2),
                ReLU(inplace=True),
                MaxPool2d(kernel_size=2, stride=2)
            ),
            Sequential(
                Conv2d(128, 256, 3, padding=2),
                ReLU(inplace=True),
                Conv2d(256, 256, 3, padding=1),
                ReLU(inplace=True),
                MaxPool2d(kernel_size=2, stride=2)
            ),
            Sequential(
                Conv2d(256, 512, 3, padding=1),
                ReLU(inplace=True),
                Conv2d(512, 512, 3, padding=1),
                ReLU(inplace=True),
                MaxPool2d(kernel_size=2, stride=2)
            ),
            Sequential(
                Conv2d(512, 512, 3, padding=1),
                ReLU(inplace=True),
                Conv2d(512, 512, 3, padding=1),
                ReLU(inplace=True),
                MaxPool2d(kernel_size=3, stride=3)
            )
        )

        self.classifier = Sequential(
            Linear(512, 512),
            ReLU(inplace=True),
            Linear(512, 512),
            ReLU(inplace=True),
            Linear(512, 10)
        )
        self.batchnorm = nn.BatchNorm2d(3)

    def forward(self, input, mode, n, indx_list, f_list):
        input = self.batchnorm(input)
        x = self.features(input, mode, 15, 15, n, indx_list, f_list)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x, mode, 15, 15, n, indx_list, f_list)
        return x


loss_func = nn.CrossEntropyLoss()

def test(model, mode, PE_size, indx_list, f_list):
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        out = model(batch_x, mode, PE_size, indx_list, f_list)
        # out = model(batch_x)
        loss = loss_func(out, batch_y)
        eval_loss += loss.data.item()*(batch_x.size()[0])

        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()

        eval_acc += num_correct.data.item()
        print(loss.data.item(), float(num_correct.data.item())/float(batch_x.size()[0]))

    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(test_data)),
                                                      float(eval_acc) / float(len(test_data))))


def run(model_path, PE_size):
    happend_list, f_list = f.systolic_array_fault_list(PE_size)
    print(happend_list, f_list)
    model = vgg_cifar10()
    model.load_state_dict(torch.load(model_path))
    mode = 'is_no_delay'
    test(model, mode, PE_size, happend_list, f_list)

for epoch in range(1):
    run("./VGG_11_para_2_keep8.pkl", 16)
