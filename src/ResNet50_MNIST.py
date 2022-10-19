# Importing the necessary libraries
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import numpy as np
from torch import optim
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statistics import mean
import datetime as datetime
import copy
import random
import time
from tqdm import tqdm
from pathlib import Path
import os
from torch.utils.tensorboard import SummaryWriter
import logging

def connect_gdrive():
  try:
      from google.colab import drive
      drive.mount('/content/drive/', force_remount=True)
      print("Note: using Google Drive")
  except:
      print("WARNING: Google Drive NOT CONNECTED")

def create_dir(path):
  if not os.path.exists(path):
    try:
      os.makedirs(path)
    except OSError:
      pass
def setup_dir(path):
  create_dir(path)
  dirs = ['model','figures','log','tensorboard_events','metrics']
  for dir in dirs:
    create_dir(f'{path}/{dir}')

def save_metrics(train_acc,val_acc,test_acc,train_loss,val_loss,test_loss):
  tea = timestamp.strftime("%H.%M_%d-%m-%Y")
  with Path(f'{save_path}/metrics/accuracy.npy').open('ab') as f:
    np.save(f, np.array([[100. * round(train_acc,4),100. * round(val_acc,4),100. * round(test_acc,4)]]))
  with Path(f'{save_path}/metrics/loss.npy').open('ab') as f:
    np.save(f, np.array([[train_loss,val_loss,test_loss]]))
def save_model(epoch):
  torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss': criterion,
              }, f'{save_path}/model/model.pth')

def train_model(model, iterator, optimizer, criterion, device):

  epoch_loss = 0
  epoch_accuracy = 0

  model.train()

  for inputs, labels in tqdm(iterator):
    inputs = inputs.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()
    predictions = model(inputs)
    loss = criterion(predictions, labels)
    loss.backward()
    optimizer.step()

    pred = predictions.argmax(1, keepdim=True) 
    accuracy_value = pred.eq(labels.view_as(pred)).sum().float() / labels.shape[0]


    epoch_loss += loss.item()
    epoch_accuracy += accuracy_value.item()

  final_epoch_loss = epoch_loss/len(iterator)
  final_epoch_acc = epoch_accuracy/len(iterator)

  return final_epoch_loss, final_epoch_acc

def validation(model, iterator, criterion, device):
  epoch_loss = 0
  epoch_accuracy = 0

  model.eval()

  with torch.no_grad():

    for inputs, labels in tqdm(iterator):

      inputs = inputs.to(device)
      labels = labels.to(device)

      val_predictions = model(inputs)
      val_loss = criterion(val_predictions, labels)


      pred = val_predictions.argmax(1, keepdim=True) 
      accuracy_value = pred.eq(labels.view_as(pred)).sum().float() / labels.shape[0]

      epoch_loss += val_loss.item()
      epoch_accuracy += accuracy_value.item()
  
  final_epoch_loss = epoch_loss/len(iterator)
  final_epoch_acc = epoch_accuracy/len(iterator)

  return final_epoch_loss, final_epoch_acc

class ResNet50(nn.Module):
  def __init__(self):
    super(ResNet50,self).__init__()
    # define model
    self.model = torchvision.models.resnet50(pretrained=False,num_classes=10)
    
  def forward(self,x):
    return self.model(x)

g_drive = False
resume_train = True

BASE_DIR = "./content/dl_cv3"
if g_drive == True:
  connect_gdrive()
  BASE_DIR = "/content/drive/MyDrive/dl_cv3"
  create_dir(BASE_DIR)
  create_dir(f'{BASE_DIR}/figures/accuracy')
  create_dir(f'{BASE_DIR}/figures/loss')

MODEL = "ResNet50" #@param ["VGG16", "ResNet50"]
DATASET_NAME = "MNIST"
BATCH_SIZE = 64 #@param ["64", "128", "256"] {type:"raw"}
MODIFIED = "MOD" #@param ["MOD", "NoMOD"]
OPTIMIZER_NAME = "ADAM" #@param ["ADAM", "SGD"]
MLR_RATE = "0.0001" #@param ["0.0001", "0.01"]
TRAIN_EPOCHS = 10#@param ["10", "50"] {type:"raw", allow-input: true}

DIR = f'{MODEL}_{DATASET_NAME}_{MODIFIED}_{OPTIMIZER_NAME}_{MLR_RATE}'
print(f"{BASE_DIR}/{DIR}")
save_path = f"{BASE_DIR}/{DIR}"
data_path = f"{BASE_DIR}/data"
setup_dir(save_path)


model = ResNet50()
model = model.cuda() # Using CUDA enabled GPU for training the model
print(model)

if (MODIFIED == "MOD"):
  trans_train = transforms.Compose([transforms.RandomResizedCrop((224)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.Grayscale(num_output_channels=3),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
else:
  trans_train = transforms.Compose([transforms.Resize((224,224)),
                                        # transforms.RandomHorizontalFlip(),
                                        transforms.Grayscale(num_output_channels=3),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])

trans = transforms.Compose([transforms.Resize((224,224)),
                                      #transforms.RandomHorizontalFlip(),
                                      transforms.Grayscale(num_output_channels=3),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))])


train_dataset = torchvision.datasets.MNIST(root=f'{data_path}/mnist', train=True, download=True, transform=trans_train)
test_dataset = torchvision.datasets.MNIST(root=f'{data_path}/mnist', train=False, download=True, transform=trans)


train_size = int(0.8 * len(train_dataset))   #80% training data
valid_size = len(train_dataset) - train_size #20% validation data
train_data, val = torch.utils.data.random_split(train_dataset, [train_size, valid_size])

val = copy.deepcopy(val)
val.dataset.transform = trans

train_loader = torch.utils.data.DataLoader(train_data, batch_size= BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val, batch_size = BATCH_SIZE, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size= BATCH_SIZE, shuffle=False)

examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)

fig = plt.figure()
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])

# te_dir = f'{save_path}/tensorboard_events/test'
# writer = SummaryWriter(te_dir)
# %load_ext tensorboard
# %tensorboard --logdir=/content/drive/MyDrive/dl_cv3/VGG16_MNIST/tensorboard_events/test

logging.basicConfig(filename=f"{save_path}/log/newlog.log",
                    format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p',
                    filemode='a')
 
# Creating an object
logger = logging.getLogger()
 
# Setting the threshold of logger to DEBUG
logger.setLevel(logging.INFO)

timestamp = datetime.datetime.now()
device = torch.device('cuda')
criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()
if (OPTIMIZER_NAME == "ADAM"):
  optimizer = optim.Adam(model.parameters(), lr=np.float16(MLR_RATE), weight_decay=0)
else:
  optimizer = torch.optim.SGD(model.parameters(), lr=np.float16(MLR_RATE), momentum=0.9) 

start_epoch = 0
epochs = TRAIN_EPOCHS 

if resume_train == True and os.path.isfile(f'{save_path}/model/model.pth'):
  checkpoint = torch.load(f'{save_path}/model/model.pth')
  model.load_state_dict(checkpoint['model_state_dict'])
  print('Previously trained model weights state_dict loaded...')
  logger.info('Previously trained model weights state_dict loaded...')
  # load trained optimizer state_dict
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  print('Previously trained optimizer state_dict loaded...')
  logger.info('Previously trained optimizer state_dict loaded...')
  old_epochs = checkpoint['epoch']
  # load the criterion
  criterion = checkpoint['loss']
  print('Trained model loss function loaded...')
  logger.info('Trained model loss function loaded...')
  print(f"Previously trained for {old_epochs+1} number of epochs...")
  logger.info(f"Previously trained for {old_epochs+1} number of epochs...")
  # train for more epochs
  start_epoch = old_epochs+1
  print(f"Train for {epochs - start_epoch} more epochs...")
  logger.info(f"Train for {epochs - start_epoch} more epochs...")

else:
  if resume_train == False:
    print("Training New Model")
    logger.info("Training New Model")
  else:
    print("Pre-Trained Model Not Found!... \n Starting New Training...")
    logger.info("Pre-Trained Model Not Found!... \n Starting New Training...")

iterations = []
train_losses = []
test_losses = []
val_losses = []
train_accuracies = []
test_accuracies = []
val_accuracies = []

epoch = start_epoch

# Training the model
since = time.time()
print('Start of training:', time.asctime(time.localtime(time.time())))
logger.info('Start of training')
while epoch<epochs:
# for epoch in range(epochs):
    timestamp = datetime.datetime.now()
    print("Date/Time stamp", timestamp)
    epoch_time = time.time()
    print('Epoch {}/{}'.format(epoch + 1, epochs))
    logger.info(f'Epoch {epoch + 1}/{epochs}')

    train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    print(f'Training accuracy: {100. * train_acc:.3f} | Training loss: {train_loss:.3f}')
    logger.info(f'Training accuracy: {100. * train_acc:.3f} | Training loss: {train_loss:.3f}')

    val_loss, val_acc = validation(model, val_loader, criterion, device)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    print(f'Validation accuracy: {100. * val_acc:.3f} | Validation loss: {val_loss:.3f}')
    logger.info(f'Validation accuracy: {100. * val_acc:.3f} | Validation loss: {val_loss:.3f}')

    test_loss, test_acc = validation(model, test_loader, criterion, device)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    print(f'Test accuracy: {100. * test_acc:.3f} | Test loss: {test_loss:.3f}')
    logger.info(f'Test accuracy: {100. * test_acc:.3f} | Test loss: {test_loss:.3f}')


    time_elapsed = time.time() - epoch_time
    print('Epoch completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info('Epoch completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('-' * 10)
    if ((epoch+1)%5 == 0):
      print('Saving Model...')
      logger.info('Saving Model...')
      save_model(epoch)
      print('Saving Finished')
      logger.info('Saving Finished')
    # writer.add_scalars('VGG16_loss', {'train_loss':train_loss,
    #                             'val_loss':val_loss,
    #                             'test_loss': test_loss}, epoch+1)
    # writer.add_scalars('VGG16_accuracy', {'train_accuracy':100. * round(train_acc,4),
    #                         'val_accuracy':100. * round(val_acc,4),
    #                         'test_accuracy': 100. * round(test_acc,4)}, epoch+1)
    # writer.add_pr_curve('pr_curve', labels, predictions, epoch+1)
    save_metrics(train_acc,val_acc,test_acc,train_loss,val_loss,test_loss)
    epoch+=1

final_time_elapsed = time.time() - since
print('Training completed in {:.0f}m {:.0f}s'.format(final_time_elapsed // 60, final_time_elapsed % 60))
logger.info('Training completed in {:.0f}m {:.0f}s'.format(final_time_elapsed // 60, final_time_elapsed % 60))
print(f'End of training{time.asctime(time.localtime(time.time()))}')
logger.info(f'End of training{time.asctime(time.localtime(time.time()))}')
print('Saving Model...')
logger.info('Saving Model...')
save_model(epoch)
print('Saving Finished')
logger.info('Saving Finished')

# Clear PyTorch cache
model = None
with torch.no_grad():
    torch.cuda.empty_cache()
torch.cuda.empty_cache()