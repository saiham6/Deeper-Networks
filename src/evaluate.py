from pathlib import Path
import numpy as np
import os
import matplotlib.pyplot as plt

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

def smooth(scalars, weight=0.45):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return np.array(smoothed)
    
def load_numpy_data(model,mod,lr,metric):
  p = Path(f'{BASE_DIR}/{model}_{DATASET_NAME}_{mod}_{OPTIMIZER_NAME}_{lr}/metrics/{metric}.npy')
  with p.open('rb') as f:
      fsz = os.fstat(f.fileno()).st_size
      out = np.load(f)
      while f.tell() < fsz:
          out = np.concatenate((out, np.load(f)),axis=0)
  return out

g_drive = False

BASE_DIR = "./content/dl_cv3"
if g_drive == True:
  connect_gdrive()
  BASE_DIR = "/content/drive/MyDrive/dl_cv3"

MODEL_1 = "VGG16" #@param ["VGG16", "ResNet50"]
MODEL_2 = "ResNet50" #@param ["VGG16", "ResNet50"]
DATASET_NAME = "Cifar"  #@param ["Cifar", "MNIST"]
BATCH_SIZE = 64 #@param ["64", "128", "256"] {type:"raw"}
MODIFIED = "MOD" #@param ["MOD", "NoMOD"]
MODIFIED_2 = "MOD" #@param ["MOD", "NoMOD"]
OPTIMIZER_NAME = "ADAM" #@param ["ADAM", "SGD"]
MLR_RATE = "0.0001" #@param ["0.0001", "0.01"]
MLR_RATE_2 = "0.0001" #@param ["0.0001", "0.01"]
TRAIN_EPOCHS = 10 #@param ["10", "50"] {type:"raw", allow-input: true}
ACCURACY_PATH = f'accuracy'
LOSS_PATH = f'loss'
METRIC = ACCURACY_PATH #@param ["ACCURACY_PATH", "LOSS_PATH"] {type:"raw"}

DIR = f'{MODEL_1}_{DATASET_NAME}_{MODIFIED}_{OPTIMIZER_NAME}_{MLR_RATE}'

out_1 = load_numpy_data(MODEL_1,MODIFIED,MLR_RATE,METRIC)
out_2 = load_numpy_data(MODEL_2,MODIFIED_2,MLR_RATE_2,METRIC)

mod = 'augmented' if (MODIFIED_2 == 'MOD') else ''
dat = 'CIFAR-10' if (DATASET_NAME == 'Cifar') else 'MNIST'
met = 'Accuracy' if (METRIC == ACCURACY_PATH) else 'Loss'

plt.plot(np.arange(start=1, stop=11, step=1), smooth(out_1[:,0][:10]), label = f"{MODEL_1}_Train_{MLR_RATE}")
plt.plot(np.arange(start=1, stop=11, step=1), smooth(out_1[:,1][:10]), label = f"{MODEL_1}_Validation_{MLR_RATE}")
plt.plot(np.arange(start=1, stop=11, step=1), smooth(out_2[:,0][:10]), label = f"{MODEL_2}_Train_{MLR_RATE_2}")
plt.plot(np.arange(start=1, stop=11, step=1), smooth(out_2[:,1][:10]), label = f"{MODEL_2}_Validation_{MLR_RATE_2}")

plt.xticks(np.arange(start=1, stop=11, step=1))
plt.xlabel('Epochs')

plt.ylabel(f'{met}')
plt.legend(loc=0)
plt.grid(True)
plt.tight_layout()
plt.title(f'{MODEL_1} and {MODEL_2} on {mod} {dat} data with {OPTIMIZER_NAME}')
plt.savefig(f'{BASE_DIR}/figures/{METRIC}/{MODEL_1}_{MODEL_2}_{MODIFIED}_{DATASET_NAME}_{OPTIMIZER_NAME}_{MLR_RATE}_{met}.png', bbox_inches='tight')
plt.show()