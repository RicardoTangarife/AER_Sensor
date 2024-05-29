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
parser.add_argument('--conf1', type=float, default=0.90, help='Nivel de confianza para enviar reconocimiento de evento acustico 1')
parser.add_argument('--conf2', type=float, default=0.74, help='Nivel de confianza para enviar reconocimiento de evento acustico 2')
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

def fused_predict_ARQ1_TL(path, input, output1, output2, interpreter, evento1, evento2, conf1, conf2):
    samplerate = 22050
    longitudMaxAudio = 4
    Nmfcc = 3
    Nfft = 4096
    NwinL = 4096
    iterableNhopL = 1.0
    NhopL =  4096       #int(iterableNhopL*NwinL)
    k_size = 2
    num_rows = Nmfcc
    num_columns = int(samplerate*longitudMaxAudio/NhopL) + int(samplerate*longitudMaxAudio/NhopL*0.05)  #Calculo longitud de salida de mfcc con 5% de tolerancia para longitud de audios
    num_channels = 1

    audio = extract_features(path, Nmfcc, Nfft, NhopL, NwinL)
    audioP = audio.reshape(1, num_rows, num_columns, num_channels)
    input_data = audioP
    #print("INPUT DATA SHAPE", input_data.shape )

    # Invoke the model on the input data
    interpreter.set_tensor(input['index'], input_data)
    interpreter.invoke()

    # Get the result
    output_data_1 = interpreter.get_tensor(output1['index'])[0]
    output_data_2 = interpreter.get_tensor(output2['index'])[0]
    #print("OUTPUT DATA SHAPE - 1", output_data_1.shape)
    #print(output_data_1)
    #print("OUTPUT DATA SHAPE - 2", output_data_2.shape)
    #print(output_data_2)

    #probOut1 = model.predict(audioP)[0]
    indexMax1 = np.argmax(output_data_1)
    indexMax2 = np.argmax(output_data_2)
    #print(output_data)
    #print(indexMax)
    maxProb1 = output_data_1[indexMax1]
    maxProb2 = output_data_2[indexMax2]

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
    else:
      print("No se identificó evento con nivel de confianza")
      save_audio(path, segments_path)
      #classP = 'None'
    #print('Class predicted :',classP,'\n\n')



#fusedModelSavePathGunScream_optAVG_1_tflite =  "models/fused_gun_scream_optAVG_1.tflite"
fusedModelSavePathGunSiren_optAVG_1_tflite =  "models/fused_gun_siren_optAVG_1.tflite"
#fusedModelSavePathScreamSiren_optAVG_1_tflite =  "models/fused_scream_siren_optAVG_1.tflite"

#interpreter = tf.lite.Interpreter(model_path = fusedModelSavePathGunScream_optAVG_1_tflite)
interpreter = tf.lite.Interpreter(model_path = fusedModelSavePathGunSiren_optAVG_1_tflite)
#interpreter = tf.lite.Interpreter(model_path = fusedModelSavePathScreamSiren_optAVG_1_tflite)

interpreter.allocate_tensors()  # Needed before execution!

input = interpreter.get_input_details()[0]  # Model has single input.
output1 = interpreter.get_output_details()[0]  # Model has double output.
output2 = interpreter.get_output_details()[1]  # Model has double output.


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
        fused_predict_ARQ1_TL(filePathSave, input, output1, output2, interpreter, "DISPARO", "SIRENA", conf1, conf2)

except KeyboardInterrupt:
    # Manejar la interrupción del teclado (Ctrl + C)
    pass

finally:
    # Limpiar la configuración de la GPIO antes de salir
    pass


