import os
from pathlib import Path

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from dataset import Dataset

batch_size = 1024
data_path = Path("./")

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.5,
                0.5,
                0.5,
            ],
            std=[0.5, 0.5, 0.5],
        ),
    ]
)

trainset = Dataset(data_dir=data_path, train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

testset = Dataset(data_dir=data_path, train=False, transform=transform)
testloader = DataLoader(testset, batch_size=1, shuffle=False)

num_images = 14846
num_classes = 100


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model = models.resnet18(num_classes=num_classes)
model = model.to(device)
cudnn.benchmark = True

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss(reduction="mean")

print("train")
model = model.train()

total, tp = 0, 0
epoch = 100
for i in range(epoch):
    acc_count = 0
    for j, (x, y) in enumerate(trainloader):
        x = x.to(device)
        y = y.to(device)

        predict = model.forward(x)
        loss = criterion(predict, y)
        pre = predict.argmax(1).to(device)
        total += y.shape[0]
        tp += (y == pre).sum().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if j % 10 == 0:
            print(
                "%03depoch, %05d, loss=%.5f, acc=%.5f" % (i, j, loss.item(), tp / total)
            )


print("test")
model = model.eval()
total, tp = 0, 0
predict_l = []
for x in testloader:
    x = x.to(device)
    predict = model.forward(x)
    pre = predict.argmax(1).to("cpu")
    predict_l.append(pre.item())

submit = pd.DataFrame(columns=["image_id", "labels"])
submit["image_id"] = list(range(500))
submit["labels"] = predict_l
submit.to_csv(data_path / Path("submition.csv"), index=False)
