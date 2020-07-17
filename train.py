#aladdinpersson
#----- version 2 -----#
#----- version 3 -----#
# jdhaskdhaskhska
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as dataset

torch.manual_seed(23)

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN,self).__init__()

        self.fc1 = nn.Linear(input_size,50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = self.fc1(x) #layer 1
        x = F.relu(x)
        x = self.fc2(x)

        return x

model = NN(784,10)

x = torch.randn(64,784)

print ('x shape is', x.shape)

print (model(x).shape)

#set device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 784
num_classes = 10
batch_size = 64
num_epoch = 1
learning_rate = 0.001

train_dataset = dataset.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True)

test_dataset = dataset.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True)

model = NN(input_size,num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epoch):
    for batch_idx, (data,targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)
        data = data.reshape(data.shape[0], -1)
        scores = model(data)
        loss = criterion(scores, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0

#this is locally chanaged fileee
    with torch.no_grad():

        for x,y in test_loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0],-1)

            scores = model(x)   #shae of scores 64x10
            _,predictions = scores.max(1) # max of scores in second dimension
            num_correct += (predictions == y).sum()

            num_samples += y.size(0)





        print (f'Got epoch is {epoch}_ {num_correct}/{num_samples} with accuracy {(float(num_correct)/float(num_samples))*100:.2f}')


        # model.train()


# check_accuracy(train_loader,model)
# check_accuracy(test_loader,model)

























