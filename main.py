import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(42)
np.random.seed(42)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(3,1)

    def forward(self, x):
        return self.fc(x)


model = MyModel()

criterion = nn.MSELoss()

learning_rate = 0.01
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

x = torch.randn(2,3)
y = torch.randn(2)

for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(x).reshape(-1)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

print(x)
print(y)
y_hat = model(x)
print(y_hat)
