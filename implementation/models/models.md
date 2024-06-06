# Models for Sensor AER Implementation

This directory contains various models designed for the implementation of the Acoustic Event Recognition (AER) sensor. The models included are trained individually through transfer learning, trained as biclass classifiers via transfer learning, or fused using top-performing fusion strategies. The table below provides a brief description of each model along with their corresponding F1-Score performance.

The models are in TensorFlow Lite format for execution on edge devices. The models cater to the three different acoustic events studied: gunshots, sirens, and screams. Additionally, they are available for the two different MFCC hyperparameter configurations studied: Configuration 1 with lower complexity and Configuration 2 with higher performance.

**Configuration 1:** (Lower Complexity (_1))
- Nfft with a value of 4096 samples.
- NwinL with a value of 4096 samples.
- NhopL with a value of 100%.
- Nmfcc with a value of 3.
- Ksize with a value of 2.

**Configuration 2:** (Higher Performance (_4))
- Nfft with a value of 4096 samples.
- NwinL with a value of 4096 samples.
- NhopL with a value of 100%.
- Nmfcc with a value of 45.
- Ksize with a value of 5.

| Model                                          |   F1-Score  | Description                                                            |
|------------------------------------------------|-------------|------------------------------------------------------------------------|
| fused_gun_scream_optAVG_1.tflite               |   98%       | Fused model for gunshot and scream with AVG method Performance-based Filter Merge Without Similarity Metric Evaluation configuration 1    |
| fused_gun_scream_optAVG_4.tflite               |   98%       | Fused model for gunshot and scream with AVG method Performance-based Filter Merge Without Similarity Metric Evaluation configuration 2    |
| fused_gun_scream_sim20_optAVG_1_N.tflite       |   98%       | Fused model for gunshot and scream with AVG method (similarity threshold 80%) with normalized filters configuration 1    |
| fused_gun_scream_sim20_optAVG_1.tflite         |   97%       | Fused model for gunshot and scream with AVG method (similarity threshold 80%) without normalized filters configuration 1    |
| fused_gun_scream_sim20_optAVG_4_N.tflite       |   98%       | Fused model for gunshot and scream with AVG method (similarity threshold 80%) with normalized filters configuration 2    |
| fused_gun_siren_optAVG_1.tflite                |   89%       | Fused model for gunshot and siren with AVG method  Performance-based Filter Merge Without Similarity Metric Evaluation configuration 1    |
| fused_gun_siren_optAVG_4.tflite                |   98%       | Fused model for gunshot and siren with AVG method  Performance-based Filter Merge Without Similarity Metric Evaluation configuration 2    |
| fused_gun_siren_sim20_optAVG_1_N.tflite        |   89%       | Fused model for gunshot and siren with AVG method (similarity threshold 80%) with normalized filters configuration 1    |
| fused_gun_siren_sim40_optAVG_1_N.tflite        |   89%       | Fused model for gunshot and siren with AVG method (similarity threshold 60%) with normalized filters configuration 1    |
| fused_gun_siren_sim40_optAVG_4_N.tflite        |   96%       | Fused model for gunshot and siren with AVG method (similarity threshold 60%) with normalized filters configuration 2    |
| fused_scream_siren_optAVG_1.tflite             |   91%       | Fused model for scream and siren with AVG method Performance-based Filter Merge Without Similarity Metric Evaluation configuration 1    |
| fused_scream_siren_optAVG_4.tflite             |   97%       | Fused model for scream and siren with AVG method  Performance-based Filter Merge Without Similarity Metric Evaluation configuration 2    |
| fused_scream_siren_sim20_optAVG_1_N.tflite     |   90%       | Fused model for scream and siren with AVG method (similarity threshold 80%) with normalized filters configuration 1    |
| saved_gun_scream_siren_TL_1.tflite             |   84%       | Transfer learning model for gunshot, scream, and siren configuration 1    |
| saved_gun_scream_siren_TL_4.tflite             |   96%       | Transfer learning model for gunshot, scream, and siren configuration 2    |
| saved_gun_scream_TL_1.tflite                   |   95%       | Transfer learning model for gunshot and scream biclass configuration 1    |
| saved_gun_scream_TL_4.tflite                   |   99%       | Transfer learning model for gunshot and scream biclass configuration 2    |
| saved_gun_siren_TL_1.tflite                    |   82%       | Transfer learning model for gunshot and siren biclass configuration 1     |
| saved_gun_siren_TL_4.tflite                    |   97%       | Transfer learning model for gunshot and siren biclass configuration 2     |
| saved_gunshot_TL_1.tflite                      |   95%       | Transfer learning model for gunshot configuration 1                       |
| saved_gunshot_TL_4.tflite                      |   99%       | Transfer learning model for gunshot configuration 2                       |
| saved_scream_siren_TL_1.tflite                 |   84%       | Transfer learning model for scream and siren biclass configuration 1      |
| saved_scream_siren_TL_4.tflite                 |   97%       | Transfer learning model for scream and siren biclass configuration 2      |
| saved_scream_TL_1.tflite                       |   99%       | Transfer learning model for scream configuration 1                        |
| saved_scream_TL_4.tflite                       |   99%       | Transfer learning model for scream configuration 2                        |
| saved_siren_TL_1.tflite                        |   82%       | Transfer learning model for siren configuration 1                         |
| saved_siren_TL_4.tflite                        |   97%       | Transfer learning model for siren configuration 2                         |
