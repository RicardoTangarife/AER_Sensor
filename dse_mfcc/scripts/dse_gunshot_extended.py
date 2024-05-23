# -*- coding: utf-8 -*-
import librosa
import pandas as pd
import os
import struct
import json
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime
from sklearn import metrics 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)
tf.get_logger().setLevel('ERROR')   #0| INFO| [Default] Print all messages #1| WARNING| Filter out INFO messages  #2| ERROR| Filter out INFO & WARNING messages  #3| NONE 


gpus = tf.config.list_physical_devices('GPU')
print(gpus)
tf.config.set_visible_devices(gpus[2], 'GPU')

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# TODO SOBRE DATASET AUDIO
#https://github.com/karolpiczak/ESC-50
#https://labs.freesound.org/datasets/
#https://zenodo.org/record/2552860#.Y21fVHaZOUk
#https://github.com/DCASE-REPO/dcase2018_baseline/tree/master/task2
#https://zenodo.org/record/3665275#.Y21fNnaZOUk
#https://arxiv.org/pdf/2010.00475.pdf

#USADA
#https://zenodo.org/record/3519845#.Y3QXLXaZOUl

# Se obtienen las diferentes rutas de los datos, tanto audios como metadata y path para obtener las muestras
file_path = '../../data/UrbanSound8K/audio'
explosions_path = '../../data/sounds-explosions/explosions'
urbansound8k = pd.read_csv('../../data/UrbanSound8K/metadata/UrbanSound8K.csv')
file_viz = glob.glob('../../data/UrbanSound8K/audio/fold1/*')
filenameOutput = '../output/DSE_gunshot.json'
modelCheckpoint_path = '../output/models/DSE_gunshot_check'
saveModels_folderpath = '../output/models/gunshot_models'


#pd.set_option('display.max_rows', None)
#urbansound8k.head()

#Se organiza dataset para seleccionar la clase gun shot de interes y las otras renombrarse como no gunshot
urbansound8k.loc[urbansound8k["class"] != "gun_shot", "class"] = "non_gun_shot"

#Se cambian los valores de la columna classID para identificar unicamente las dos clases de interes
urbansound8k.loc[urbansound8k["classID"] != 6, "classID"] = 1
urbansound8k.loc[urbansound8k["classID"] == 6, "classID"] = 0

#urbansound8k.head()

#se cuenta el numero total de muestras de la clase de interes 
totalSamplesGunshot =  urbansound8k.loc[((urbansound8k["class"]=="gun_shot"))].count()[0]
print("Total Samples Gunshot", totalSamplesGunshot)

#se carga el numero total de muestras de la clase a agregar al dataset - explosions
totalSamplesExplosions = 0
lastNameFilesExplosions=[]
for file in os.listdir(explosions_path):
    if file.endswith('.wav'):
        lastNameFilesExplosions.append(file)
        totalSamplesExplosions+=1
print("Total Samples Explosions", totalSamplesExplosions)

#Se crea dataframe con los datos de los audios de sonidos de la clase a agregar - explosions
dfExplosion = pd.DataFrame(lastNameFilesExplosions, columns=['slice_file_name'])
dfExplosion['classID'] = 1
dfExplosion['class'] = 'non_gun_shot'
dfExplosion['fold'] = 0

#dfExplosion

#Agregar en un dataset todos los sonidos de interes con la clase nueva agregada - explosions
print(urbansound8k.count()[0])
dfUrbansound8kV2 = pd.concat([urbansound8k, dfExplosion],)
#dfUrbansound8kV2 = urbansound8k.append(dfExplosion)
print("Total Samples", dfUrbansound8kV2.count()[0])

#se toma una muestra aleatoria del tamaño de las muestras de la clase de interes para que quede balanceado
#urbansound8k[urbansound8k['class'] == "non_gun_shot"].sample(n=totalSamplesGunshot).head()

#se toma una muestra aleatoria del tamaño de las muestras de la clase de interes para que quede balanceado 
dfNonGunShot = urbansound8k[urbansound8k['class'] == "non_gun_shot"].sample(n=totalSamplesGunshot)
dfGunShot = urbansound8k[urbansound8k['class'] == "gun_shot"]

#se unen en un solo dataset las clases de interes con muestras balanceadas
dfComplete = pd.concat([dfNonGunShot, dfGunShot],)
#dfComplete = dfNonGunShot.append(dfGunShot)
print("Balanced Total Samples", dfComplete.count()[0])

print("Balanced Total Samples GunShot", dfComplete[dfComplete['class'] == "gun_shot"].count()[0])

#Se cambia el dataset para unicamente tomar las clases de interés para el trabajo
#dfComplete.head()


"""Extracting features using Librosa"""

#Se define funcion para extrar las caracteristicas con la librería librosa, obetiendo los coeficientes ceptrales de frecuencia de Mel
#Se realiza un pading en el tamaño para que concuerden los tamaños de las caracteristicas de entrada al modelo.

def extract_features(file_name, Nmfcc, Nfft, NhopL, NwinL):
    
    samplerate = 22050
    longitudMaxAudio = 4
    max_pad_len = int(samplerate*longitudMaxAudio/NhopL) + int(samplerate*longitudMaxAudio/NhopL*0.05)  #Calculo longitud de salida de mfcc con 5% de tolerancia para longitud de audios

    try:
      audio, sample_rate = librosa.load(file_name, res_type='soxr_hq') 
      mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=Nmfcc, n_fft=Nfft, hop_length=NhopL, win_length=NwinL)
      pad_width = max_pad_len - mfccs.shape[1]
      mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

    except Exception as e:
      print("Error encountered while parsing file: ", file_name)
      return None 
    #print(mfccs.shape)
    return mfccs

#Se realiza la extracción de caracteristicas, teniendo en cuenta la clase, si el sonido es de la carpeta agregada de la clase explosions va y busca este sonido en la carpeta requerida

def get_features(Nmfcc, Nfft, NhopL, NwinL):
  features = []
  # Iterate through each sound file and extract the features 
  for index, row in dfComplete.iterrows():
      if(row["fold"]!=0):
          file_name = os.path.join(os.path.abspath(file_path),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
          #file_name = ""
      elif(row["fold"]==0):
          file_name = os.path.join(os.path.abspath(explosions_path),str(row["slice_file_name"]))
          #file_name = ""
      #print(file_name)
      class_label = row["classID"]
      data = extract_features(file_name, Nmfcc, Nfft, NhopL, NwinL)
      
      features.append([data, class_label])

  # Convert into a Panda dataframe 
  featuresdf = pd.DataFrame(features, columns=['feature','class_label'])
  return featuresdf

def splitFeaturesTrainTest(featuresdf, num_rows, num_columns, num_channels):
  X = np.array(featuresdf.feature.tolist())
  y = np.array(featuresdf.class_label.tolist())

  # Encode the classification labels
  le = LabelEncoder()
  yy = to_categorical(le.fit_transform(y)) 

  # split the dataset 
  from sklearn.model_selection import train_test_split 

  x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 3)

  x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)
  x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)
  num_labels = yy.shape[1]
  return x_train, x_test, y_train, y_test, num_labels

"""#Creating the Model"""

# Constructing model with RELu and SoftMax activation functions:
# Constructing model with RELu and SoftMax activation functions:
def getModel(num_rows, num_columns, num_channels, num_labels, k_size):
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=k_size, input_shape=(num_rows, num_columns, num_channels), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=32, kernel_size=k_size, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=64, kernel_size=k_size, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=128, kernel_size=k_size, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Dropout(0.2))
    model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    model.add(Dense(num_labels, activation='softmax'))
    return model

"""#DSE GUNSHOT EXTENDED"""

import subprocess

bashCommand = "nvidia-smi -L"
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
outputGPU, error = process.communicate()
specGPU = outputGPU.split(b'\n')[2].decode("utf-8")
print("SpecsGPU:", specGPU)

def my_grep(pattern, file):
  string_file = file.decode("utf-8")
  for line in string_file.split('\n'):
      if pattern in line:
          return line

bashCommand = "lscpu"
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
specCPU, error = process.communicate()
specCore = my_grep('Core(s) per socket:',specCPU)
print("SpecsCore:", specCore)

specCPU = my_grep('Model name:', specCPU)
print("SpecsCPU:", specCPU)

#Model and History container
models = []
histories = []
reports = []
cmatrixs = []
trainAcc = []
testAcc = []
trainTimes = []
numberEpochsRan = []

NExp = 1              #1              #Identificación con número de experimento
samplerate = 22050
longitudMaxAudio = 4
valuesNmfcc = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45]   #[3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45]  #Valores de parametro a variar para el numero de coeficientes MFCC
valuesNfft = [256, 512, 1024, 2048, 4096]     #[256, 512, 1024, 2048, 4096] #Valores de parametro a variar para la longitud de la FFT
valuesWinL = [256, 512, 1024, 2048, 4096]      #[256, 512, 1024, 2048, 4096]  #Valores de parametro a variar para el tamaño de ventana, este debe ser menor o igual a NFFT, la función hace padding con 0
valuesHopL = [0.25, 0.5, 0.75, 1.0]            #[0.25, 0.5, 0.75, 1.0]      #Valores de parametro a variar para el overlaping opuesto de hop_length
valuesKernelSize = [2, 3, 5, 7]                 #[2, 3, 5, 7] #Valores de parametro de tamaño de kernel a variar dentro del modelo


for Nmfcc in valuesNmfcc:                     #Loop para variar valores del parametro n_mfcc => Numero de coeficientes MFCC
  for Nfft in valuesNfft:                         #Loop para variar valores del parametro n_fft => Tamaño del la FFT
    for iterableNwinL in valuesWinL:              #Loop para variar valores del parametro Win_Length => Longitud de la ventana de muestreo
      if iterableNwinL<=Nfft:
        NwinL = iterableNwinL
      else:
        continue
      for iterableNhopL in valuesHopL:            #Loop para variar valores del parametro Hop_Length => 1/Overlaping
        #if (Nfft==2048 and NwinL<=512 and iterableNhopL==0.25): #or (Nfft==2048 and NwinL==512 and iterableNhopL==0.5):
          #continue
        NhopL = int(iterableNhopL*NwinL)
        num_rows = Nmfcc
        num_columns = int(samplerate*longitudMaxAudio/NhopL) + int(samplerate*longitudMaxAudio/NhopL*0.05)  #Calculo longitud de salida de mfcc con 5% de tolerancia para longitud de audios
        num_channels = 1
        print(f'EXERIMENT NUMBER = {NExp}')
        print(f'N_MFCC= {Nmfcc}, Nfft= {Nfft}, NwinL= {NwinL}, NhopL= {NhopL}')

        startP = datetime.now()
        featuresdf = get_features(Nmfcc, Nfft, NhopL, NwinL)
        durationPreprocessing = datetime.now() - startP
        totalSamples = int(dfComplete.count()[0])
        durationAvgPreprocs = durationPreprocessing/totalSamples
        
        print('\nPreprocessing Finished For:\n\n')
        print(f'N_MFCC= {Nmfcc}, Nfft= {Nfft}, NwinL= {NwinL}, NhopL= {NhopL}')
        print('\n\nPreprocessing Duration Average Per Sample: \n\n', durationAvgPreprocs)

        x_train, x_test, y_train, y_test, num_labels = splitFeaturesTrainTest(featuresdf, num_rows, num_columns, num_channels)

        
        for k_size in valuesKernelSize:           #Loop para variar valores del parametro kernel size => Tamaño del kernel de capas convolucionales
          #if (Nfft==4096 and NwinL==2048 and iterableNhopL==0.5 and k_size<=3):
            #continue
          models = []
          histories = []
          reports = []
          cmatrixs = []
          trainAcc = []
          testAcc = []
          trainTimes = []
          numberEpochsRan = []
          for i in range(5):                        #Loop para promediar el experimento realizandolo 5 veces con los mismos valores de parametros
            model = getModel(num_rows, num_columns, num_channels, num_labels, k_size)
            #Compile model with categorical crossentropy loss function 
            model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
            # Generate a print
            print('------------------------------------------------------------------------')
            print(f'Training for N_MFCC= {Nmfcc}, Nfft= {Nfft}, NwinL= {NwinL}, NhopL= {NhopL}, Ksize= {k_size}, ...')

            #Train Model
            num_epochs = 80
            num_batch_size = 256
            
            earlystopper = EarlyStopping(patience=10, verbose=0)
            checkpointer = ModelCheckpoint(filepath= modelCheckpoint_path, verbose=0, save_best_only=True)
            start = datetime.now()
            history = model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data = (x_test, y_test), callbacks=[earlystopper, checkpointer], verbose=0)
            duration = datetime.now() - start
            print("Training completed in time: ", duration)
            trainTimes.append(duration)

            # Evaluating the model on the training and testing set
            score = model.evaluate(x_train, y_train, verbose=0)
            print("Training Accuracy: ", score[1])
            trainAcc.append(score[1])

            score = model.evaluate(x_test, y_test, verbose=0)
            print("Testing Accuracy: ", score[1])
            testAcc.append(score[1])


            y_true = np.argmax(y_test,axis=1)
            startPredicTime = datetime.now()
            y_pred = np.argmax(model.predict(x_test),axis=1)
            durationPredicTime = datetime.now() - startPredicTime
            
            totalPredicSamples = int(x_test.shape[0])
            durationAvgPredicTime  = durationPredicTime/totalPredicSamples
            print(f"Prediction completed in AVG time for {totalPredicSamples} samples: ", durationAvgPredicTime)
            print('\nConfusion Matrix :\n\n')
            print(confusion_matrix(y_true,y_pred))
            print('\n\nClassification Report : \n\n',classification_report(y_true,y_pred))

            # Add trained model, histoy, and reports to container
            numberEpochsRan.append(len(history.history['loss']))
            models.append(model)
            histories.append(history)
            reports.append(classification_report(y_true,y_pred))
            cmatrixs.append(confusion_matrix(y_true,y_pred))

            """#Save the entire model as a SavedModel: Saved_Gunshot_NExp_RepetitionNumber."""
            model.save(saveModels_folderpath+f"/Saved_Gunshot_NExp{NExp}_Rep{i+1}")

            # Convert the model to tf lite
            converter = tf.lite.TFLiteConverter.from_saved_model(saveModels_folderpath+f"/Saved_Gunshot_NExp{NExp}_Rep{i+1}") # path to the SavedModel directory
            tflite_model = converter.convert()

            # Save the model.
            with open(saveModels_folderpath+f"/Saved_Gunshot_NExp{NExp}_Rep{i+1}_lite.tflite", 'wb') as f:
              f.write(tflite_model)

          Experiment = {
            'NExp': NExp,
            'N_MFCC': Nmfcc,
            'Nfft': Nfft,
            'NwinL': NwinL,
            'NhopL': NhopL,
            'Ksize': k_size,
            'durationAvgPreprocs': durationAvgPreprocs,
            #'models': models,
            #'histories': histories,
            'reports': reports,
            'cmatrixs': cmatrixs,
            'trainAcc': trainAcc,
            'testAcc': testAcc,
            'trainTimes': trainTimes,
            'numberEpochsRan': numberEpochsRan,
            'avgPredicTime': durationAvgPredicTime,
            'specGPU': specGPU,
            'specCPU': specCPU,
            'specCore': specCore
          }
          
          print(Experiment)
          json_object = json.dumps(Experiment, ensure_ascii = False, indent=2, sort_keys=True, default=str)
          print(json_object)

          # 1. Read json file
          with open(filenameOutput, "r") as f:
              data = json.load(f)
              f.close() 
          # 2. Update json object
          data["Experiments"].append(Experiment)
          # 3. Write json file
          with open(filenameOutput, "w") as f:
              json.dump(data, f, ensure_ascii = False, indent=2, sort_keys=True, default=str)
              f.close()

          NExp += 1 #Add one to experiments counter


