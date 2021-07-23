# 本节主要学习基于最小二乘法的多项式与曲面拟合.

# 直接使用PyTorch了.

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

weights = torch.FloatTensor([0.5, 3, 2.4]).unsqueeze(1)
bias = torch.FloatTensor([0.9])


def make_features(x):
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, 4)], 1)


def get_batch(batch_size=32):
    random = torch.randn(batch_size)
    x = make_features(random)
    y = x.mm(weights) + bias[0]

    data = x.cuda(), y.cuda()

    return data


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.main = nn.Linear(3, 1)

    def forward(self, x):
        out = self.main(x)
        return out


def main():
    model = Model().cuda()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-4)

    for _ in range(32768):
        inputs, target = get_batch()
        output = model(inputs)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    data = np.linspace(-1, 1, 30)
    inputs = torch.from_numpy(data)
    inputs = inputs.unsqueeze(1)
    inputs = torch.cat([inputs ** i for i in range(1, 4)], 1)
    inputs = inputs.float()
    prediction = inputs.mm(weights) + bias[0]
    y_predict = model(inputs.cuda())
    plt.plot(data, prediction.numpy(), "ro", data, y_predict.data.cpu().numpy())
    plt.legend(["input data", "prediction"])
    plt.show()


if __name__ == "__main__":
    main()
