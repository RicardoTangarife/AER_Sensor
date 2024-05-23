## Design Space Exploration for MFCC Feature Extraction

The Mel-frequency cepstral coefficients (MFCC) have been widely employed due to their proven effectiveness in describing audio features and structures. This method has proven valuable in automatic speech recognition, music information retrieval, environmental sound retrieval, and the detection, classification, and recognition of acoustic events.

In the context of sound, representing it through MFCC encapsulates the key characteristics of the sound. MFCC allows us to form two-dimensional matrices that can be interpreted as images, leveraging the advantages of CNNs for processing such data. The efficacy of CNNs in Acoustic Event Recognition (AER) has been demonstrated in the results of the community challenges of Detection and Classification of Acoustic Scenes and Events (DCASE) over the past decade.

### MFCC Feature Extraction Process

The process of calculating MFCC unfolds as follows:

1. **Segmentation and Window Application**: The signal is divided into short frames.
2. **Power Spectrum Computation**: Using the Fast Fourier Transform (FFT) for each window.
3. **Mel Scale Transformation**: The power spectrum is transformed to the Mel scale by applying a Mel filter bank.
4. **Logarithm of Filter Bank Energies**: The total energy of each filter bank is calculated, and its logarithm is obtained.
5. **Discrete Cosine Transform (DCT)**: The MFCC coefficients are extracted and organized into a matrix.

![image](https://github.com/RicardoTangarife/AER_Sensor/assets/36963665/f937035a-df90-4d31-95c9-0e1f56aca429)


### Hyperparameter Optimization for MFCC

To optimize the feature extraction of MFCC for the recognition of selected urban acoustic events, we conducted a preliminary exploration of the design space to understand the influence of MFCC parameters. We evaluated the accuracy performance of CNN models for these acoustic recognition tasks. The hyperparameters included in the design space exploration and the values evaluated were as follows:

- **Length of the Fast Fourier Transform (FFT) (Nfft)**: Values of 256, 512, 1024, 2048, and 4096 samples.
- **Window Size (NwinL)**: Values of 256, 512, 1024, 2048, and 4096 samples.
- **Window Step Length (NhopL)**: Variations of 25%, 50%, 75%, and 100%.
- **Number of MFCC (Nmfcc)**: Ranging from 3 to 45 with an increment of 3.
- **CNN Kernel Size (Ksize)**: Values of 2, 3, 5, and 7.

We used the Hanning window as the default window type, as it has been reported to exhibit the best overall performance without being tied to a specific application.

The ranges of hyperparameter values evaluated were selected based on previous studies conducted on MFCC feature configurations. This combination of five hyperparameters resulted in a total of 3600 possible models, excluding configurations where the FFT size is greater than the window size.

In our study, we identified the Pareto optimal models for recognizing the acoustic events of gunshots, sirens, and screams using metrics such as the model's performance measured by the F1-Score and the model's inference computation requirements measured in FLOPS. Generally, models using higher parameter values (such as MFCC coefficients, window size, smaller hops, etc.) tend to perform better in acoustic event recognition. However, these models are more computationally expensive and may take longer to train and evaluate.

![image](https://github.com/RicardoTangarife/AER_Sensor/assets/36963665/147ab22e-63fa-40ed-a8df-9a78c637846d)


### Repository Structure

This part of repository is organized as follows:

- **scripts**: Contains the different scripts for executing model experiments under various MFCC hyperparameter configurations, including scripts for execution and temporal measurements.
- **output**: Includes the output of the scripts, such as execution times and model performance metrics for each configuration.
- **data_analyzer**: Contains scripts for analyzing the output from the experiments. The results of these analyses are found in the `output_data` directory within this folder.
