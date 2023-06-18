import os
import csv
import cv2
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import matplotlib.pyplot as plt 
# In[]
# 可以不要用
for dirname, _, filenames in os.walk('kaggle/input'):
    for filename in filenames[:3]:
        print(os.path.join(dirname, filename))
    if len(filenames) > 3:
        print("...")
# In[]
TRAIN_PATH = "kaggle/input/captcha-hacker-2023-spring/dataset/train"
TEST_PATH = "kaggle/input/captcha-hacker-2023-spring/dataset/test"
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# In[]
alphabets = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789*"
alphabets2index = {alphabet:i for i, alphabet in enumerate(alphabets)}
# In[]
class Task1Dataset(Dataset):
    def __init__(self, data, root, return_filename=False,transform=None):
        self.folders = ["task1", "task2", "task3"]
        self.data = [sample for sample in data if sample[0].startswith(tuple(self.folders))]
        self.return_filename = return_filename
        self.root = root
        self.transform=transform
        self.classes = sorted(list(set([sample[1] for sample in self.data])))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
    def __getitem__(self, index):
        filename, label = self.data[index]
        img = cv2.imread(f"{self.root}/{filename}")
        img = cv2.resize(img, (96, 96))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        img = cv2.medianBlur(img, 3)
        img[img==255]=1
        if self.return_filename:
            return torch.FloatTensor(img), filename
        if len(label)==1:
            label1=label[0]
            label2='*'
            label3='*'
            label4='*'
        elif len(label)==2:
            label1=label[0]
            label2=label[1]
            label3='*'
            label4='*'
        else:
            label1=label[0]
            label2=label[1]
            label3=label[2]
            label4=label[3]
        return torch.FloatTensor(img), torch.tensor([alphabets2index[label1],alphabets2index[label2],alphabets2index[label3],alphabets2index[label4]])
    def __len__(self):
        return len(self.data)
# In[]
class Bidirectional(nn.Module):
    def __init__(self, inp, hidden, out, lstm=True):
        super(Bidirectional, self).__init__()
        self.rnn = nn.LSTM(inp, hidden, bidirectional=True)
        self.embedding = nn.Linear(hidden*2, out)
    def forward(self, X):
        recurrent, _ = self.rnn(X)
        out = self.embedding(recurrent)     
        return out
    
class EfficientNet(nn.Module):
    def __init__(self):
        super(EfficientNet, self).__init__()
        pretrain=models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)

        pretrain._modules['avgpool']=nn.Identity()
        pretrain._modules['classifier']=nn.Identity()
        print(pretrain)
        self.conv1 = nn.Conv2d(1, 3, kernel_size=(3, 3), stride=1, padding=1)
        self.efficient=pretrain
        
        self.fc1=nn.Linear(1280,256)

        #self.bn1 = nn.BatchNorm1d(256),
        self.rnn = Bidirectional(256, 1024, 63+1)
        del pretrain
        
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs=self.efficient(outputs)
        print(outputs.shape)
        N, C, w, h = outputs.size()
        print(outputs.shape)
        outputs = outputs.view(N, -1, h)
        print(outputs.shape)
        outputs = outputs.permute(0, 2, 1)
        print(outputs.shape)
        outputs=self.fc1(outputs)
        print(outputs.shape)
        outputs = outputs.permute(1, 0, 2)
        outputs = self.rnn(outputs)


        return outputs
# In[]
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        pretrain=models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)#pretrain._modules['conv1']
        self.bn1 = pretrain._modules['bn1']
        self.relu = pretrain._modules['relu']
        self.maxpool =pretrain._modules['maxpool']
        self.layer1 = pretrain._modules['layer1']
        self.layer2 = pretrain._modules['layer2']
        self.layer3 = pretrain._modules['layer3']
        self.layer4 = pretrain._modules['layer4']
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1=nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(pretrain._modules['fc'].in_features,63)
            )
        self.fc2=nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(pretrain._modules['fc'].in_features,63)
            )
        self.fc3=nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(pretrain._modules['fc'].in_features,63)
            )
        self.fc4=nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(pretrain._modules['fc'].in_features,63)
            )
        del pretrain
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs)
        outputs = self.relu(outputs)
        outputs = self.maxpool(outputs)
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.layer4(outputs)
        outputs = self.avgpool(outputs)
        output1 = self.fc1(outputs)
        output2 = self.fc2(outputs)
        output3 = self.fc3(outputs)
        output4 = self.fc4(outputs)
        return output1,output2,output3,output4
# In[]
def plot_correct(training,valid,epochs):
    plt.plot(range(1,epochs+1),training) 
    plt.plot(range(1,epochs+1), valid)    
    plt.title('task1_EfficientNet with pretrain weight')
    plt.ylabel('Accuracy(%)'), plt.xlabel('Epochs')
    plt.legend(['Train(with pretraining)','Valid(with pretraining)'])
    plt.show()
def plot_loss(training,valid,epochs):
    plt.plot(range(1,epochs+1),training) 
    plt.plot(range(1,epochs+1), valid)    
    plt.title('task1_EfficientNet with pretrain weight')
    plt.ylabel('Loss(%)'), plt.xlabel('Epochs')
    plt.legend(['Train(with pretraining)','Valid(with pretraining)'])
    plt.show()
# In[]
def train(model ,loss_func, optimizer, epochs, train_dataset, valid_dataset):
    training_acc=[]
    valid_acc=[]
    training_loss=[]
    valid_loss=[]
    best_acc=0.0
    for epoch in range(epochs):
        all_train_correct=0
        # Train model
        model.train()
        sample_count = 0
        for i, (inputs, labels) in enumerate(train_dataset):
            # 1.Define variables
            inputs=inputs.view(-1,1,96,96).to(device)
            labels=labels.to(device).long()
            label1=labels[:,0]
            label2=labels[:,1]
            label3=labels[:,2]
            label4=labels[:,3]
            #labels=labels.cuda().long()
            # 2.Forward propagation
            #outputs=model.forward(inputs=inputs)

            outputs=model.forward(inputs=inputs)
            print(outputs)
            T = outputs.size(0)
            N = outputs.size(1)
        
            input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.int32)
            target_lengths = torch.full(size=(N,), fill_value=4, dtype=torch.int32)
            #outputs=model(inputs)
            # 3.Clear gradients
            optimizer.zero_grad()
            # 4.Calculate softmax and cross entropy loss
            loss = loss_func(outputs, labels, input_lengths, target_lengths)
            # 5. Calculate gradients
            loss.backward()
            # 6. Update parameters
            optimizer.step()
            # 7.Total correct predictions
            pred1 = torch.max(output1.data, 1)[1] #14
            pred2 = torch.max(output2.data, 1)[1] #14
            pred3 = torch.max(output3.data, 1)[1] #14
            pred4 = torch.max(output4.data, 1)[1] #14

            pred=torch.stack([pred1, pred2, pred3, pred4], dim=0)
            pred = torch.transpose(pred, 0, 1)
            #all_train_correct += (pred == labels).long().sum()

            all_train_correct += (torch.all(pred == labels, dim=1)).long().sum()
            sample_count +=len(inputs)
        training_loss.append(loss.cpu())
        training_acc.append((100*all_train_correct/sample_count).cpu())
        print(f'For training epoch:{epoch+1} training acc:{training_acc[-1]}')
        # Test model
        model.eval()
        sample_count = 0
        all_test_correct=0
        with torch.no_grad():
            for inputs, labels in valid_dataset:
                inputs=inputs.view(-1,1,96,96).to(device)
                labels=labels.to(device).long()
                label1=labels[:,0]
                label2=labels[:,1]
                label3=labels[:,2]
                label4=labels[:,3]
                #outputs=model.forward(inputs=inputs)
                output1,output2,output3,output4=model.forward(inputs=inputs)
                #pred = torch.max(outputs.data, 1)[1]
                loss1=loss_func(output1,label1)
                loss2=loss_func(output2,label2)
                loss3=loss_func(output3,label3)
                loss4=loss_func(output4,label4)
                loss=loss1+loss2+loss3+loss4
                
                pred1 = torch.max(output1.data, 1)[1] #14
                pred2 = torch.max(output2.data, 1)[1] #14
                pred3 = torch.max(output3.data, 1)[1] #14
                pred4 = torch.max(output4.data, 1)[1] #14
                pred=torch.stack([pred1, pred2, pred3, pred4], dim=0)
                pred = torch.transpose(pred, 0, 1)
                #all_test_correct += (pred == labels).long().sum()
                all_test_correct += (torch.all(pred == labels, dim=1)).long().sum()
                sample_count +=len(inputs)
            valid_loss.append(loss.cpu())
            now_testing_acc=(100* all_test_correct/sample_count)
            valid_acc.append((100* all_test_correct/sample_count).cpu())
            print(f'For valid epoch:{epoch+1} valid acc:{valid_acc[-1]}')
            print()
            if now_testing_acc>best_acc:
                best_acc=now_testing_acc
                torch.save(model.state_dict(),'weight/CTC_EfficientNet.pt')
    torch.cuda.empty_cache()
    return training_acc,valid_acc, training_loss,valid_loss,best_acc
# In[]
train_data = []
val_data = []

with open(f'{TRAIN_PATH}/annotations.csv', newline='') as csvfile:
    for row in csv.reader(csvfile, delimiter=','):
        if random.random() < 0.8:
            train_data.append(row)
        else:
            val_data.append(row)
# In[]

train_ds = Task1Dataset(train_data, root=TRAIN_PATH)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

val_ds = Task1Dataset(val_data, root=TRAIN_PATH)
val_dl = DataLoader(val_ds, batch_size=32, shuffle=False)


# In[]
model=Model().to(device)
epochs=300
lr=1e-3
#optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#optimizer=torch.optim.SGD(model.parameters(), lr=lr,momentum=0.9,weight_decay=5e-4)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
loss_func = nn.CrossEntropyLoss()

training_acc,valid_acc, training_loss,valid_loss,best_acc=train(model,loss_func, optimizer, epochs, train_dl, val_dl)
print(best_acc)
# In[]
model=EfficientNet().to(device)
epochs=600
lr=1e-3
#optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#optimizer=torch.optim.SGD(model.parameters(), lr=lr,momentum=0.9,weight_decay=5e-4)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
loss_func = nn.CTCLoss()

training_acc,valid_acc, training_loss,valid_loss,best_acc=train(model,loss_func, optimizer, epochs, train_dl, val_dl)
print(best_acc)
# In[]
plot_correct(training_acc,valid_acc,epochs)

a=torch.tensor(training_loss, requires_grad=True)
b=torch.tensor(valid_loss, requires_grad=True)
plot_loss(a.detach().numpy(),b.detach().numpy(),epochs)
# In[]
test_data = []
with open(f'{TEST_PATH}/../sample_submission.csv', newline='') as csvfile:
    for row in csv.reader(csvfile, delimiter=','):
        test_data.append(row)

test_ds = Task1Dataset(test_data, root=TEST_PATH, return_filename=True)
test_dl = DataLoader(test_ds, batch_size=1, drop_last=False, shuffle=False)


if os.path.exists('submission.csv'):
    csv_writer = csv.writer(open('submission.csv', 'a', newline=''))
else:
    csv_writer = csv.writer(open('submission.csv', 'w', newline=''))
    csv_writer.writerow(["filename", "label"])

model.load_state_dict(torch.load('weight/dropout75_EfficientNet.pt'))
model.eval()
with torch.no_grad():
    for image, filenames in test_dl:
        
        image = image.view(-1,1,96,96).to(device)
        
        pred1,pred2,pred3,pred4 = model(image)
        pred1 = torch.argmax(pred1, dim=1)
        pred2 = torch.argmax(pred2, dim=1)
        pred3 = torch.argmax(pred3, dim=1)
        pred4 = torch.argmax(pred4, dim=1)
        if filenames[0][:5]=="task1":
            csv_writer.writerow([filenames[0], alphabets[pred1.item()]])
        elif filenames[0][:5]=="task2":
            if pred2==62:
                csv_writer.writerow([filenames[0], alphabets[pred1.item()]+'I'])
            else:
                csv_writer.writerow([filenames[0], alphabets[pred1.item()]+alphabets[pred2.item()]])
        else:
            csv_writer.writerow([filenames[0], alphabets[pred1.item()]+alphabets[pred2.item()]+alphabets[pred3.item()]+alphabets[pred4.item()]])