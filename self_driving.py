# Import Libraries
from sklearn.model_selection import RandomizedSearchCV
import torch.nn.functional as F
import torch
import cv2
import numpy
import os
import glob
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch import optim
from torchsummary import summary
import matplotlib.pyplot as plt
from torch.autograd import Variable

random_seed = 3116
torch.manual_seed(random_seed)
# Setting Up the model


class Convnet(torch.nn.Module):

    def __init__(self):
        super(Convnet, self).__init__()
        self.normalization = nn.BatchNorm2d(3)
        self.convlayer1 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=0),
            nn.ReLU())

        self.convlayer2 = nn.Sequential(
            nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=0),
            nn.ReLU())

        self.convlayer3 = nn.Sequential(
            nn.Conv2d(36, 48, kernel_size=5, stride=2, padding=0),
            nn.ReLU())

        self.convlayer4 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU())

        self.convlayer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU())

        #self.vectorized = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(1152, 1164, bias=True)
        self.fc2 = nn.Linear(1164, 100, bias=True)
        self.fc3 = nn.Linear(100, 50, bias=True)
        self.fc4 = nn.Linear(50, 10)
        self.vehicle_control = nn.Linear(10, 1)

    def forward(self, out):
        out = self.normalization(out)
        out = self.convlayer1(out)
        out = self.convlayer2(out)
        out = self.convlayer3(out)
        out = self.convlayer4(out)
        out = self.convlayer5(out)
        #out = out.view(out.size(0), -1)
        out = out.reshape(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = self.vehicle_control(out)

        return out

        # return F.softmax(out, dim=1)


model = Convnet()
# model

if torch.cuda.is_available():
    model.cuda()

summary(model, (3, 66, 200))

# Loading the Dataset
img_path = 'C:/Users/Ejimofor/Documents/driving_dataset'
angles_path = 'C:/Users/Ejimofor/Documents/driving_dataset/angles.csv'


class Dataset(Dataset):
    def __init__(self):

        self.filepath = img_path
        self.steerangles = pd.read_csv(
            self.filepath+'/angles.txt', sep=' ', header=None)
        self.steerangles.columns = ['Image_ID', 'Steer_angles']
        self.steerangles.to_csv(angles_path, index=None)
        self.steer_angles = pd.DataFrame(
            self.steerangles, index=None, columns=None)

        self.image_ID = self.steer_angles.iloc[:, 0]
        self.Steer_Ang = self.steer_angles.iloc[:, 1]

        # incase data transformations are to be done
        #self.transform = transform

    def __len__(self):
        return len(self.image_ID)

    def __getitem__(self, index):

        img = cv2.imread(self.filepath + '/' + self.image_ID[index])
        img = cv2.resize(img, (66, 200), interpolation=cv2.INTER_AREA)

        steer_labels = self.Steer_Ang[index]
        steer_rad_labels = (steer_labels * np.pi)/180
        steer_rad_labels = torch.tensor(steer_rad_labels)

        return (torch.from_numpy(img).float(), steer_labels)


# Initializing and Splitting the dataset
data = Dataset()
total_len = data.__len__()
train_split = int(0.44708 * total_len)
valid_split = int(0.3327 * total_len)
test_split = total_len - train_split - valid_split
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
    data, (train_split, valid_split, test_split))


print(total_len)
print(train_split)
print(valid_split)
print(test_split)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=200, shuffle=True)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=200, shuffle=True)
#test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=200, shuffle=True)

train_iter = iter(train_loader)
valid_iter = iter(valid_loader)
#Loss, optimizer and training
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

loss_function = torch.nn.MSELoss()


def normalize(X):  # minmaxscaler
    min_X = min(X)
    max_X = max(X)
    return (X - min_X)/(max_X - min_X)


model.train()
for epoch in range(100):

    train_iter = iter(train_loader)
    valid_iter = iter(valid_loader)

    running_loss = 0.0
    train_counter = 0
    trainloss_history = []

    val_running_loss = 0.0
    valloss_history = []
    val_counter = 0

    for i in range(0, len(train_loader)):
        images, labels = next(train_iter)

        images, labels = images, labels.unsqueeze(-1).float()
        images, labels = images, normalize(labels)

        images, labels = images, labels

        optimizer.zero_grad()

        outputs = model(images.permute(0, 3, 1, 2))

        # RMSE Loss
        loss = torch.sqrt(loss_function(outputs, labels))

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_counter += 1
        trainloss_history.append(running_loss)

        if i % 100 == 0:
            print("Epoch: {}, count: {}, Training loss: {}".format(
                epoch, i+1, running_loss/train_counter))

        running_loss = 0.0
        train_counter = 0

    for i in range(0, len(valid_loader)):
        val_images, val_labels = next(valid_iter)

        val_images, val_labels = val_images, val_labels.unsqueeze(-1).float()
        val_images, val_labels = val_images, normalize(val_labels)

        val_images, val_labels = val_images, val_labels

        with torch.no_grad():

            outputs = model(val_images.permute(0, 3, 1, 2))

            loss = loss_function(outputs, val_labels)

            val_running_loss += loss.item()
            val_counter += 1
            valloss_history.append(val_running_loss)

        if i % 100 == 0:
            print("Epoch: {}, count: {}, validation loss: {}".format(
                epoch, i+1, val_running_loss/val_counter))

        val_running_loss = 0.0
        val_counter = 0

print('Finished Training and Validation')

# RMSE for the test set of images

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=200, shuffle=True)
test_iter = iter(test_loader)


for epoch in range(1):
    total = 0.0
    correct = 0.0
    count = 0

    for i in range(0, len(test_loader)):
        test_images, test_labels = next(test_iter)
        test_images, test_labels = test_images, test_labels.unsqueeze(
            -1).float()
        test_images, test_labels = test_images.cuda(), normalize(test_labels).cuda()

        model.eval()
        with torch.no_grad():
            if torch.cuda.is_available():
                test_images = test_images.cuda()
            output = model(test_images.permute(0, 3, 1, 2)).cuda()
            out = torch.argmax(output, 1)[0]

            total += test_labels.size(0)*test_labels.size(1)
            correct += (out.cpu() == test_labels.cpu()).sum()
            acc = correct.double()/total * 100

        count += 1

        if i % 10 == 0:
            print("epoch: {} , i: {},  accuracy: {}".format(epoch, i+1, acc))

        count = 0


train_iter = iter(train_loader)
ti = []
#ti = np.array(ti)
for i in range(0, len(train_loader)):
    i, j = next(train_iter)
    ti.append((i, j))


print(len(ti))
# Hyperparameter Tuning, Regularization with ImageTransformations
parameter_grid = {'batch_size': [list(range(50, 500, 50))], 'SGDRegressor_eta0': [0.003, 0.007, 0.01, 0.03, 0.06, 0.1], 'SGDRegressor_alpha': [
    0.0001, 0.003, 0.005], 'lr': [1e-3, 1e-4, 2e-3, 2e-4], 'SGDRegressor_penalty': ['2']}
cnn_search = RandomizedSearchCV(model.train(), parameter_grid, cv=5, n_iter=5)
cnn_search.fit(ti)

# Mixup implementation
model.train()
for epoch in range(2):

    train_iter = iter(train_loader)
    valid_iter = iter(valid_loader)

    running_loss = 0.0
    train_counter = 0
    trainloss_history = []

    val_running_loss = 0.0
    valloss_history = []
    val_counter = 0
    alpha = 1

    for i in range(0, len(train_loader)):
        images, labels = next(train_iter)

        images, labels = images, labels.unsqueeze(-1).float()
        images, labels = images, normalize(labels)

        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()

            lam = numpy.random.beta(alpha, alpha)
            mimages = Variable(lam * images + (1. - lam) * images)
            mlabels = Variable(lam * labels + (1. - lam) * labels)

        optimizer.zero_grad()

        outputs = model(mimages.permute(0, 3, 1, 2)).cuda()

        loss = loss_function(outputs, mlabels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_counter += 1
        trainloss_history.append(running_loss)

        if i % 100 == 0:
            print("Epoch: {}, count: {}, Training loss: {}".format(
                epoch+1, i+1, running_loss/train_counter))

        running_loss = 0.0
        train_counter = 0
