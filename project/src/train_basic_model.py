#!/usr/bin/env python
# coding: utf-8

# In[1]:
import cv2
from PIL import Image
from  matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset
import os
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F

# In[2]:


# Checking if cuda supported GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# In[3]:


os_path = './data/widerface/train/images/'
label_txt = os_path[:-7] + 'label.txt'


# In[4]:


os_path = './data/widerface/train/images/'
def get_list_from_filenames(file_path):
    images_paths =[]
    labels,key = [], ""
    with open(label_txt, 'r') as fr:
        test_dataset = fr.read().split("\n")
    for i, each_line in enumerate(test_dataset):
        if(len(each_line)>0 and each_line[0] == '#'):
            images_paths.append(each_line[2:])
            key = each_line[2:]
        else:
            if each_line!='':
                labels.append((key,np.array(each_line.split()).astype('float32'))) 
    return images_paths,labels
images_paths,labels = get_list_from_filenames(os_path)


# In[7]:


def get_ignored_params(model):
    # Generator function that yields ignored params.
    b = [model.conv1, model.bn1, model.fc_finetune]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_non_ignored_params(model):
    # Generator function that yields params that will be optimized.
    b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_fc_params(model):
    # Generator function that yields fc layer params.
    b = [model.fc_yaw, model.fc_pitch, model.fc_roll]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param

def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)


# In[8]:


class NeuralNet(nn.Module):
    def __init__(self, block, layers, num_bins):
        self.inplanes = 64
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_yaw = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch = nn.Linear(512 * block.expansion, num_bins)
        self.fc_roll = nn.Linear(512 * block.expansion, num_bins)

        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        pre_yaw = self.fc_yaw(x)
        pre_pitch = self.fc_pitch(x)
        pre_roll = self.fc_roll(x)

        return pre_yaw, pre_pitch, pre_roll


# In[22]:

# Creating customised Image dataset to be used for dataloader
class WiderDataset(Dataset):
    def __init__(self,data_dir,transform,img_n_labels):
        self.data_dir = data_dir
        self.transform = transform
        self.X_y = img_n_labels
        self.image_mode = 'G'
        self.length = len(img_n_labels)

    def __len__(self):
        return self.length

    
    # face_x face_y face_width face_height landmark1.x landmark1.y 0.0 
    # landmark2.x landmark2.y 0.0 landmark3.x landmark3.y 0.0 landmark4.x 
    # landmark4.y 0.0landmark5.x landmark5.y 0.0 confidence pitch yaw roll
    def __getitem__(self, idx):
        if self.image_mode == 'G':
            img = Image.open(self.data_dir + self.X_y[idx][0])
        else:
            img = Image.open(self.data_dir + self.X_y[idx][0])
        label_info = self.X_y[idx][1]
        
        face_x, face_y, face_width, face_height = label_info[0], label_info[1], label_info[2], label_info[3]
        
        img = img.crop((int(face_x), int(face_y), int(face_x+face_width), int(face_y+face_height)))
        imag_name = self.data_dir + self.X_y[idx][0]
        #print("loading image:",imag_name)
        pitch = label_info[-3]
        yaw = label_info[-2]
        roll = label_info[-1]
        
        landmark1 = label_info[4:6]
        landmark2 = label_info[7:9]
        landmark3 = label_info[10:12]
        landmark4 = label_info[13:15]
        landmark5 = label_info[16:18]
        # Bin values
        bins = np.array(range(-99, 102, 3))
        binned_pose = np.digitize([yaw, pitch, roll], bins) - 1
        labels = binned_pose
        cont_labels = torch.FloatTensor([yaw, pitch, roll])
        if self.transform is not None:
            img = self.transform(img)
        return img,labels, cont_labels, self.X_y[idx][0]+str(idx)
    
if __name__ == '__main__':


    # In[10]:


    model = NeuralNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)


    # In[11]:


    load_filtered_state_dict(model, model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))


    # In[12]:

    # replace .cpu() to .cuda(0)
    print('Loading data..')
    from torchvision import transforms
    transformations = transforms.Compose([transforms.Resize(240),
    transforms.RandomCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    model.cpu()
    criterion = nn.CrossEntropyLoss().cpu()
    reg_criterion = nn.MSELoss().cpu()
    lr=0.001
    alpha = 0.001
    num_epochs = 1
    softmax = nn.Softmax(dim=1).cpu()
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cpu()

    optimizer = torch.optim.Adam([{'params': get_ignored_params(model), 'lr': 0},
                                  {'params': get_non_ignored_params(model), 'lr': lr},
                                  {'params': get_fc_params(model), 'lr': lr * 5}],
                                   lr = lr)
    train_labels, valid_labels = train_test_split(labels, test_size=0.8)
    valid_labels, test_labels = train_test_split(valid_labels, test_size=0.8)
    print("Train size: {}".format(len(train_labels)))
    print("Test size: {}".format(len(valid_labels)))
    print("Validation size: {}".format(len(test_labels)))



    # Converting the images and labels into datasets
    train_dataset=WiderDataset(os_path,transformations,train_labels)
    validation_dataset=WiderDataset(os_path,transformations,valid_labels)
    test_dataset=WiderDataset(os_path,transformations,test_labels)




    # Converting datasets into Dataloader objects (num_workers is set to 0 because setting it to a value >0 is resulting in timeout exception from the os process when device type is cuda and the datasets are custom datasets)
    batch=100


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch,
                                                   shuffle=True,
                                                   num_workers=2)
    valid_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                                   batch_size=batch,
                                                   shuffle=True,
                                                   num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                   batch_size=batch,
                                                   shuffle=True,
                                                   num_workers=2)




    for epoch in range(num_epochs):
        for i, (images, labels, cont_labels, name) in enumerate(train_loader):
            images = Variable(images).cpu()

            # Binned labels
            label_yaw = Variable(labels[:,0]).cpu()
            label_pitch = Variable(labels[:,1]).cpu()
            label_roll = Variable(labels[:,2]).cpu()

            # Continuous labels
            label_yaw_cont = Variable(cont_labels[:,0]).cpu()
            label_pitch_cont = Variable(cont_labels[:,1]).cpu()
            label_roll_cont = Variable(cont_labels[:,2]).cpu()

            # Forward pass
            yaw, pitch, roll = model(images)

            # Cross entropy loss
            loss_yaw = criterion(yaw, label_yaw)
            loss_pitch = criterion(pitch, label_pitch)
            loss_roll = criterion(roll, label_roll)

            # MSE loss
            yaw_predicted = softmax(yaw)
            pitch_predicted = softmax(pitch)
            roll_predicted = softmax(roll)

            yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 99
            pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 99
            roll_predicted = torch.sum(roll_predicted * idx_tensor, 1) * 3 - 99

            loss_reg_yaw = reg_criterion(yaw_predicted, label_yaw_cont)
            loss_reg_pitch = reg_criterion(pitch_predicted, label_pitch_cont)
            loss_reg_roll = reg_criterion(roll_predicted, label_roll_cont)

            # Total loss
            loss_yaw += alpha * loss_reg_yaw
            loss_pitch += alpha * loss_reg_pitch
            loss_roll += alpha * loss_reg_roll

            loss_seq = [loss_yaw, loss_pitch, loss_roll]
            grad_seq = [torch.ones(1)[0].cpu() for _ in range(len(loss_seq))]
            optimizer.zero_grad()
            torch.autograd.backward(loss_seq, grad_seq)
            optimizer.step()

            print ('Epoch [%d/%d], Iter [%d/%d] Losses: Yaw %.4f, Pitch %.4f, Roll %.4f'
                       %(epoch+1, num_epochs, i+1, len(train_dataset)//100, loss_yaw.data, loss_pitch.data, loss_roll.data))

        # Save models at numbered epochs.
        if epoch % 1 == 0 and epoch < num_epochs:
            print('Taking snapshot...')
            torch.save(model.state_dict(),
            'output/snapshots/' + "temp" + '_epoch_'+ str(epoch+1) + '.pkl')





