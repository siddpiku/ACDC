import torch
import numpy as np
import os
import segmentation_models_pytorch as smp
model = smp.create_model("Unet", "sam-vit_b", encoder_weights="sa-1b", encoder_depth=4, decoder_channels=[256, 128, 64, 32])
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchio as tio
import os
import matplotlib.pyplot as plt
import json
import torch.nn as nn
from torchinfo import summary
import adabound
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
from torch.optim import Adam,SGD

def get_trainbatch(traindir,labdir,batch_size):
    np.random.seed(0)
    #np.random.shuffle(traindir)
    return [(traindir[i:i + batch_size],labdir[i:i + batch_size]) for i in range(0, len(traindir), batch_size)]
device = 'cuda:0'
datadir = '~/ACDC/database/training/'
np.random.seed(0)
for a,b,c in os.walk(datadir):
    break
tr = []
lb = []
for i in b:
    for q,fi,qw in os.walk(datadir+i+'/'):
        break
    for l in qw:
        if 'frame' in l:
            if 'gt' not in l:
                tr.append(datadir+i+'/'+l)
                lb.append(datadir+i+'/'+l[:-7]+'_gt'+'.nii.gz')
class diceloss(torch.nn.Module):
    def init(self):
        super(diceLoss, self).init()
    def forward(self,pred, target):
        smooth = 1.
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )
class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1
        num = targets.size(0)
        """
               I am assuming the model does not have sigmoid layer in the end. if that is the case, change torch.sigmoid(logits) to simply logits
        """
        probs = logits#torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        
        
        # adding TV regularization
        tv_h = ((logits[:,:,1:,:] - logits[:,:,:-1,:]).pow(2)).sum()
        tv_w = ((logits[:,:,:,1:] - logits[:,:,:,:-1]).pow(2)).sum()    
       
        return score #+ 0.001*(tv_h + tv_w)
train = tr
num_epochs = 1000
batch_size = 1
#warnings.filterwarnings("ignore")
#optimizer = adabound.AdaBound(vgg3d.parameters(), lr=0.4,weight_decay = 0.004,eps = 1e-4, final_lr=0.01)
#optimizer = SGD(model.parameters(), lr=.01,momentum=0.9)#,momentum=0.9,weight_decay=0.0005)#SGD(vgg3d.parameters(), lr=0.01,momentum=0.9,weight_decay=0.0005)
optimizer = Adam(model.parameters(), lr=.001)
criterion = SoftDiceLoss()#diceloss()#nn.BCEWithLogitsLoss() #nn.CrossEntropyLoss() ### momentum=0.9 diceloss()#

if torch.cuda.is_available():
    model = model.cuda(device = device)
    criterion = criterion.cuda(device = device)
val_losses = []
train_losses_ep = []
trans = tio.CropOrPad((3,1024,1024))
transout = tio.CropOrPad((1,1024,1024))

tRESCALE = tio.RescaleIntensity((-1,1))
for epoch in range(num_epochs):
    train_losses = []
    for batch,lab in get_trainbatch(train,lb,int(batch_size)):
        y_train = []
        for cnt,i in enumerate(batch):
            if cnt == 0:     
                orig = tio.ScalarImage(batch[cnt]).data.permute(3,0,1,2)
                y_train = tio.ScalarImage(lab[cnt]).data.permute(3,0,1,2) ==2
                
                orig = tRESCALE(trans(orig))
                y_train = transout(y_train)
            else:
                temp = tio.ScalarImage(batch[cnt]).data.permute(3,0,1,2)
                temp_lab = tio.ScalarImage(lab[cnt]).data.permute(3,0,1,2) == 2
                
                temp = tRESCALE(trans(temp))
                temp_lab = transout(temp_lab)
                
                #print(temp_lab.shape,temp.shape,orig.shape,y_train.shape)
                y_train = torch.cat((y_train,temp_lab),dim = 0)
                orig = torch.cat((orig,temp),dim = 0)
        
        
        orig = orig.float().cuda(device = device)
        y_train = y_train.float().cuda(device = device)
        
        temp_num = np.random.choice(orig.shape[0],2,replace = False)
        
        orig = orig[temp_num,:,:,:]
        y_train = y_train[temp_num,:,:,:]
        
        
        # clearing the Gradients of the model parameters
        optimizer.zero_grad()

        output_train = model(orig)
        loss_train = criterion(output_train, y_train) 
        # computing the updated weights of all the model parameters
        loss_train.backward()
        optimizer.step()
        train_losses.append(loss_train.item())
        del orig
        del y_train
    train_losses_ep.append(np.mean(train_losses))
    #if epoch%(num_epochs/25) == 0:
        # printing the validation loss
    print('Epoch : ',epoch+1, '\t', 'val loss :', '\t', 'train loss : ', np.mean(train_losses))
torch.save(model.state_dict(),'/home/m256149/Documents/LeftVentricleSegmentation/code/unet-sam.pt')