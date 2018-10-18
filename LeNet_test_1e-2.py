import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import torch.utils.data.dataloader as Data
from Conv2D import Conv2d
from Linear import Linear
from mnist import MNIST


batch_size = 128

test_data = MNIST(
    '../MNIST', train=False, transform=torchvision.transforms.ToTensor(), download=True
)
print(len(test_data))

test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = Conv2d(1, 6, 5, padding=2)
        # 32->14
        self.conv2 = Conv2d(6, 16, 5)
        # 14->5
        self.conv3 = Conv2d(16, 120, 5)
        # 5->1
        self.dense1 = Linear(120, 84)
        self.dense2 = Linear(84, 10)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.batchnorm = nn.BatchNorm2d(1)

    def forward(self, input, prob, num_bits):
        input = self.batchnorm(input)
        conv1_out = self.pool(self.relu(self.conv1(input, prob, num_bits)))
        conv2_out = self.pool(self.relu(self.conv2(conv1_out, prob, num_bits)))
        conv3_out = self.relu(self.conv3(conv2_out, prob, num_bits))
        dense_in = conv3_out.view(conv3_out.size()[0], -1)
        dense_out = self.relu(self.dense1(dense_in, prob, num_bits))
        out = self.dense2(dense_out, prob, num_bits)
        return out


loss_func = nn.CrossEntropyLoss()


def test(model, prob, num_bits):
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        out = model(batch_x, prob, num_bits)
        # out = model(batch_x)
        loss = loss_func(out, batch_y)
        eval_loss += loss.data.item() * (batch_x.size()[0])

        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()

        eval_acc += num_correct.data.item()
        print(loss.data.item(), float(num_correct.data.item()) / float(batch_x.size()[0]))

    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(test_data)),
                                                  float(eval_acc) / float(len(test_data))))


def run(para_path):
    print(para_path)
    model = Net2()
    model.load_state_dict(torch.load(para_path))
    prob = 0.1
    num_bits = 7
    test(model, prob, num_bits)


for epoch in range(1):
    run('../change_Lossfunction/LeNet_L2Loss_1e-2.pkl')
