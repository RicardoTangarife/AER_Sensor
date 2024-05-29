import os
import glob
import librosa
import numpy as np
import tensorflow as tf
import sounddevice as sd
from scipy.io.wavfile import read, write
from datetime import datetime, timedelta
import time
import wiringpi as GPIO
import argparse 

# Import local functions
from audio_processing import prepare_audio, extract_features
from storage_manager import ensure_storage_space, delete_old_files, save_audio


# Argument parser
parser = argparse.ArgumentParser(description="Script para detección de eventos de audio")
parser.add_argument('--sd_path', type=str, default="/mnt/sdcard", help='Ruta para guardar el archivo de audio temporal')
parser.add_argument('--conf1', type=float, default=0.97, help='Nivel de confianza para enviar reconocimiento de evento acustico 1')
parser.add_argument('--conf2', type=float, default=0.97, help='Nivel de confianza para enviar reconocimiento de evento acustico 2')
parser.add_argument('--conf3', type=float, default=0.80, help='Nivel de confianza para enviar reconocimiento de evento acustico 3')
parser.add_argument('--max_size_segments', type=float, default=2.0, help='Tamaño máximo en GB para la carpeta de segmentos')
parser.add_argument('--max_size_events', type=float, default=1.0, help='Tamaño máximo en GB para la carpeta de eventos')
parser.add_argument('--days_to_keep', type=int, default=0, help='Número de días para mantener datos almacenados después de alcanzar la capacidad máxima')

args = parser.parse_args()

# Storage Parameters
#filePathSave = "/mnt/sdcard/tempAudio.wav"
#max_segments_size_gb = 0.045  # Maximum size in GB for the segments folder
#max_events_size_gb = 1.0      # Maximum size in GB for the events folder
#days_to_keep_storage = 0       # Number of days to keep stored data after reaching maximum capacity

# Storage Parameters from arguments
filePathSave = args.sd_path+"/tempAudio.wav"
max_segments_size_gb = args.max_size_segments
max_events_size_gb = args.max_size_events
days_to_keep_storage = args.days_to_keep

# Confidence Levels from arguments
conf1 = args.conf1
conf2 = args.conf2
conf3 = args.conf3
segments_folder = "segments"
events_folder = "events"

# Initialize GPIO configuration
GPIO.wiringPiSetup()

OUTPUT_PIN_DISPARO = 16  # Physical pin number: 37 (PIN.H4)
OUTPUT_PIN_SIRENA = 2    # Physical pin number: 15 (PIN.H6)
OUTPUT_PIN_GRITO = 3     # Physical pin number: 16 (PIN.H7)

GPIO.pinMode(OUTPUT_PIN_DISPARO, GPIO.OUTPUT)
GPIO.digitalWrite(OUTPUT_PIN_DISPARO, GPIO.LOW)
GPIO.pinMode(OUTPUT_PIN_SIRENA, GPIO.OUTPUT)
GPIO.digitalWrite(OUTPUT_PIN_SIRENA, GPIO.LOW)
GPIO.pinMode(OUTPUT_PIN_GRITO, GPIO.OUTPUT)
GPIO.digitalWrite(OUTPUT_PIN_GRITO, GPIO.LOW)



fs = 44100  # Sample rate mic Logitech diadema
fs = 48000  # Sample rate mic UMIK-1
fs = 48000  # Sample rate mic ML1
#fs = 22050  # Sample rate dataset
seconds = 4  # Duration of recording


#sd.default.samplerate = fs
#sd.default.channels = 1

#print("----------------------record device list---------------------")
#print(sd.query_devices())
sd.default.device = 5
print("-------------------------------------------------------------")


seed = 42
#tf.random.set_seed(seed)
np.random.seed(seed)


#1	3	4096	4096	4096	0	2
#2	12	4096	4096	4096	0	3
#3	27	4096	4096	4096	0	3
#4	45	4096	4096	4096	0	5

def ind_predict_ARQ4_TL(path, input1, output1, input2, output2, input3, output3, interpreter1, interpreter2, interpreter3, evento1, evento2, evento3, conf1, conf2, conf3):
    samplerate = 22050
    longitudMaxAudio = 4
    Nmfcc = 45
    Nfft = 4096
    NwinL = 4096
    iterableNhopL = 1.0
    NhopL =  4096       #int(iterableNhopL*NwinL)
    k_size = 5
    num_rows = Nmfcc
    num_columns = int(samplerate*longitudMaxAudio/NhopL) + int(samplerate*longitudMaxAudio/NhopL*0.05)  #Calculo longitud de salida de mfcc con 5% de tolerancia para longitud de audios
    num_channels = 1

    audio = extract_features(path, Nmfcc, Nfft, NhopL, NwinL)
    audioP = audio.reshape(1, num_rows, num_columns, num_channels)
    input_data = audioP
    #print("INPUT DATA SHAPE", input_data.shape )


    # Invoke the model on the input data
    interpreter1.set_tensor(input1['index'], input_data)
    interpreter1.invoke()
    # Get the result
    output_data_1 = interpreter1.get_tensor(output1['index'])[0]

    # Invoke the model on the input data
    interpreter2.set_tensor(input2['index'], input_data)
    interpreter2.invoke()
    # Get the result
    output_data_2 = interpreter2.get_tensor(output2['index'])[0]

    # Invoke the model on the input data
    interpreter3.set_tensor(input3['index'], input_data)
    interpreter3.invoke()
    # Get the result
    output_data_3 = interpreter3.get_tensor(output3['index'])[0]

    #print("OUTPUT DATA SHAPE", output_data_1.shape)
    #print(output_data_1)
    #print(output_data_2)
    #print(output_data_3)

    #probOut1 = model.predict(audioP)[0]
    indexMax1 = np.argmax(output_data_1)
    indexMax2 = np.argmax(output_data_2)
    indexMax3 = np.argmax(output_data_3)
    #print(output_data)
    #print(indexMax)
    maxProb1 = output_data_1[indexMax1]
    maxProb2 = output_data_2[indexMax2]
    maxProb3 = output_data_3[indexMax3]

    if(maxProb1>conf1 and indexMax1==1):
      print("Se identificó evento con nivel de confianza, EVENTO: ", evento1)
      print("set output pin 37 is High level")
      GPIO.digitalWrite(OUTPUT_PIN_DISPARO, GPIO.HIGH)
      GPIO.delay(2000)
      print("set output pin 37 is Low level")
      GPIO.digitalWrite(OUTPUT_PIN_DISPARO, GPIO.LOW)
      save_audio(path, events_path, evento1)
      #classP = 'gun_shot'
      #Conectar con Plataforma de datos y reportar evento
    elif(maxProb2>conf2 and indexMax2==1):
      print("Se identificó evento con nivel de confianza, EVENTO: ", evento2)
      print("set output pin 15 is High level")
      GPIO.digitalWrite(OUTPUT_PIN_SIRENA, GPIO.HIGH)
      GPIO.delay(2000)
      print("set output pin 15 is Low level")
      GPIO.digitalWrite(OUTPUT_PIN_SIRENA, GPIO.LOW)
      save_audio(path, events_path, evento2)
      #classP = 'siren'
      #Conectar con Plataforma de datos y reportar evento
    elif(maxProb3>conf3 and indexMax3==1):
      print("Se identificó evento con nivel de confianza, EVENTO: ", evento3)
      print("set output pin 16 is High level")
      GPIO.digitalWrite(OUTPUT_PIN_GRITO, GPIO.HIGH)
      GPIO.delay(2000)
      print("set output pin 16 is Low level")
      GPIO.digitalWrite(OUTPUT_PIN_GRITO, GPIO.LOW)
      save_audio(path, events_path, evento3)
      #classP = 'scream'
      #Conectar con Plataforma de datos y reportar evento
    else:
      print("No se identificó evento con nivel de confianza")
      save_audio(path, segments_path)
      #classP = 'None'
    #print('Class predicted :',classP,'\n\n')



fusedModelSavePathGun_TL_4_tflite =  "models/saved_gunshot_TL_4.tflite"
fusedModelSavePathSiren_TL_4_tflite =  "models/saved_siren_TL_4.tflite"
fusedModelSavePathScream_TL_4_tflite =  "models/saved_scream_TL_4.tflite"


interpreterGun = tf.lite.Interpreter(model_path = fusedModelSavePathGun_TL_4_tflite)
interpreterSiren = tf.lite.Interpreter(model_path = fusedModelSavePathSiren_TL_4_tflite)
interpreterScream = tf.lite.Interpreter(model_path = fusedModelSavePathScream_TL_4_tflite)


interpreterGun.allocate_tensors()  # Needed before execution!
interpreterSiren.allocate_tensors()  # Needed before execution!
interpreterScream.allocate_tensors()  # Needed before execution!

inputGun = interpreterGun.get_input_details()[0]  # Model has single input.
outputGun = interpreterGun.get_output_details()[0]  # Model has double output.

inputSiren = interpreterSiren.get_input_details()[0]  # Model has single input.
outputSiren = interpreterSiren.get_output_details()[0]  # Model has double output.

inputScream = interpreterScream.get_input_details()[0]  # Model has single input.
outputScream = interpreterScream.get_output_details()[0]  # Model has double output.

try:
    while True:
        #Grabar Audio
        print("\n\n----------------------Recording... ---------------------")
        myrecording = sd.rec(int(seconds * fs),samplerate=fs, channels=1)
        sd.wait()  # Wait until recording is finished
        write(filePathSave, fs, myrecording)  # Save as WAV file 
        print("----------------------Audio File Saved---------------------")

        prepare_audio(filePathSave)

        print("----------------------Predicción Con Audio Grabado... ---------------------")
        segments_path, events_path = ensure_storage_space("/mnt/sdcard", max_segments_size_gb, max_events_size_gb, segments_folder, events_folder, days_to_keep_storage)
        ind_predict_ARQ4_TL(filePathSave, inputGun, outputGun, inputSiren, outputSiren, inputScream, outputScream, interpreterGun, interpreterSiren, interpreterScream, "DISPARO", "SIRENA", "GRITO", conf1, conf2, conf3)

except KeyboardInterrupt:
    # Manejar la interrupción del teclado (Ctrl + C)
    pass

finally:
    # Limpiar la configuración de la GPIO antes de salir
    pass


