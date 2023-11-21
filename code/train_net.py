import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
from torchsummary import summary
import time
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import matplotlib
matplotlib.use('Agg')

# 清空显存
torch.cuda.empty_cache()

data_transforms = transforms.Compose([
    transforms.Resize([112, 112]),
    transforms.ToTensor()
])


BATCH_SIZE = 256
learning_rate = 0.001
EPOCHS = 30
numClasses = 43

# 训练集
train_data_path = "../datasets/train"
train_data = torchvision.datasets.ImageFolder(
    root=train_data_path, transform=data_transforms)

# 0.8 train, 0.2 val
ratio = 0.8
n_train_examples = int(len(train_data) * ratio)
n_val_examples = len(train_data) - n_train_examples

train_data, val_data = data.random_split(
    train_data, [n_train_examples, n_val_examples])

print(f"Number of training samples = {len(train_data)}")
print(f"Number of validation samples = {len(val_data)}")

# 直方图 training && validation data
train_hist = [0] * numClasses
for i in train_data.indices:
    tar = train_data.dataset.targets[i]
    train_hist[tar] += 1

val_hist = [0]*numClasses
for i in val_data.indices:
    tar = val_data.dataset.targets[i]
    val_hist[tar] += 1

plt.bar(range(numClasses), train_hist, label="train")
plt.bar(range(numClasses), val_hist, label="val")
legend = plt.legend(loc='upper right', shadow=True)
plt.title("Distribution Plot")
plt.xlabel("Class ID")
plt.ylabel("# of examples")

plt.savefig("./saved_images/train_val_split.png", bbox_inches='tight', pad_inches=0.5)

# dataloader
train_loader = data.DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
val_loader = data.DataLoader(val_data, shuffle=True, batch_size=BATCH_SIZE)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# AlexNet
class Alexnet(nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=192,
                      kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=192, out_channels=384,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=384, out_channels=256,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256*7*7, 1000),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),
            nn.Linear(in_features=1000, out_features=256),
            nn.ReLU(inplace=True),

            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        x = self.features(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x, h


model = Alexnet(numClasses)
print(f'The model has {count_parameters(model):,} trainable parameters')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 定义优化器&损失函数
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# CUDA
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

# print model
print(model)

# print summary of the model for the given dimension of the image
print(summary(model, (3, 112, 112)))


# print model's state dict
print("Model's state dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
print("")

print("Optimizer details:")
print(optimizer)
print("")

# calculate accuracy
def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


# training ...
def train(model, loader, opt, criterion):
    epoch_loss = 0
    epoch_acc = 0

    # train the model
    model.train()

    for (images, labels) in loader:
        images = images.cuda()
        labels = labels.cuda()

        opt.zero_grad()

        output, _ = model(images)
        loss = criterion(output, labels)

        # backpropagation
        loss.backward()

        # 计算acc
        acc = calculate_accuracy(output, labels)

        # 优化权重
        opt.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(loader), epoch_acc / len(loader)

# eval...
def evaluate(model, loader, opt, criterion):
    epoch_loss = 0
    epoch_acc = 0

    # evaluate the model
    model.eval()

    with torch.no_grad():
        for (images, labels) in loader:
            images = images.cuda()
            labels = labels.cuda()

            # 预测
            output, _ = model(images)
            loss = criterion(output, labels)

            # 计算acc
            acc = calculate_accuracy(output, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(loader), epoch_acc / len(loader)

# training and val loss and accuracies
train_loss_list = [0]*EPOCHS
train_acc_list = [0]*EPOCHS
val_loss_list = [0]*EPOCHS
val_acc_list = [0]*EPOCHS

for epoch in range(EPOCHS):
    print("Epoch-%d: " % (epoch))

    train_start_time = time.monotonic()
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    train_end_time = time.monotonic()

    val_start_time = time.monotonic()
    val_loss, val_acc = evaluate(model, val_loader, optimizer, criterion)
    val_end_time = time.monotonic()

    train_loss_list[epoch] = train_loss
    train_acc_list[epoch] = train_acc
    val_loss_list[epoch] = val_loss
    val_acc_list[epoch] = val_acc

    print("Training: Loss = %.4f, Accuracy = %.4f, Time = %.2f seconds" %
          (train_loss, train_acc, train_end_time - train_start_time))
    print("Validation: Loss = %.4f, Accuracy = %.4f, Time = %.2f seconds" %
          (val_loss, val_acc, val_end_time - val_start_time))
    print("")

# save model
MODEL_FOLDER = "../checkpoints"
if not os.path.isdir(MODEL_FOLDER):
    os.mkdir(MODEL_FOLDER)

PATH_TO_MODEL = MODEL_FOLDER + "/tsr_alexnet_30epochs.pth"
if os.path.exists(PATH_TO_MODEL):
    os.remove(PATH_TO_MODEL)
torch.save(model.state_dict(), PATH_TO_MODEL)

print("Model saved at %s" % (PATH_TO_MODEL))

_, axs = plt.subplots(1, 2, figsize=(15, 5))

# loss plot
axs[0].plot(train_loss_list, label="train")
axs[0].plot(val_loss_list, label="val")
axs[0].set_title("Plot - Loss")
axs[0].set_xlabel("Epochs")
axs[0].set_ylabel("Loss")
legend = axs[0].legend(loc='upper right', shadow=False)

# accuracy plot
axs[1].plot(train_acc_list, label="train")
axs[1].plot(val_acc_list, label="val")
axs[1].set_title("Plot - Accuracy")
axs[1].set_xlabel("Epochs")
axs[1].set_ylabel("Accuracy")
legend = axs[1].legend(loc='center right', shadow=True)

plt.savefig("./saved_images/loss_acc_plot.png", bbox_inches='tight', pad_inches=0.5)
