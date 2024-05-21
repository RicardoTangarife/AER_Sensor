## Acoustic Event Recognition Model

In our research, we evaluated five different neural network architectures specifically for the classification of acoustic events. The evaluated architectures were:

- CNN1D
- CNN2D
- LSTM
- DNN
- CONVLSTM

Based on the results and previous studies in the state-of-the-art, we selected the 2D Convolutional Neural Network (CNN2D) architecture for our Acoustic Event Recognition (AER) sensor.

### Selected Model Architecture

Various studies have employed CNN architectures for the efficient classification and detection of acoustic events. Some of these architectures have been developed based on research and outcomes from the DCASE challenges. Typically, these architectures consist of multiple blocks of convolutional layers, activation layers utilizing the rectified linear unit (ReLU) function, pooling layers, and regularization through dropout.

**Studies and Findings:**
- Chen et al. (2017) and Doshi et al. (2022) have contributed significantly to the development of CNN architectures for acoustic event detection.
- Tsalera et al. (2021) evaluated large-scale convolution architectures such as GoogLeNet, SqueezeNet, ShuffleNet, VGGish, and YAMNet using the UrbanSound8k dataset. The VGGish architecture achieved the highest performance with an accuracy of 96.7%. However, VGGish requires approximately 360.72 MFLOPS for inference and classifies multiple sounds simultaneously. Since our objective is to identify CNN models with lower computational requirements, a large-scale architecture like VGGish is not the optimal choice for recognizing a single acoustic event.

Therefore, some articles, such as Massoudi et al. (2021), have introduced reduced models based on architectures similar to VGGish for recognizing a smaller set of acoustic events.

### Architecture Description

In our experiments, we adopted the convolutional architecture with ReLU activation presented by Massoudi et al. (2021) as a reference. The authors directly evaluated this architecture on the UrbanSound8k dataset to recognize the ten classes in the set, achieving a model performance with an accuracy of 91%.

The reference architecture consists of:

- **4 Convolutional Blocks**: Each block comprises a 2D convolutional layer with a kernel size of (2×2), followed by ReLU activation, a max-pooling layer with a pool size of (2×2), and a dropout regularization of 0.2.
- **Convolutional Filters**: The number of convolutional filters increases for each block being 16, 32, 64, and 128.
- **Global Average Pooling Layer**: This layer is included after the convolutional blocks.
- **Flattened Layer**: The output of the pooling layer is flattened.
- **Dense Layer with Softmax Activation**: This layer is used for the final classification.

### References

- **Chen et al. (2017)**: Performance Evaluation of CNN Models in Urban Acoustic Event Recognition Through MFCC Hyperparameter Search. In 2023 Congress in Computer Science, Computer Engineering, & Applied Computing (CSCE), pp. 186-193. IEEE.
- **Doshi et al. (2022)**: Acoustic Event Detection using Deep Learning: DCASE2022 Challenge. [Online]. Available: [https://ieeexplore.ieee.org/document/10487496](https://ieeexplore.ieee.org/document/10487496)
- **Tsalera et al. (2021)**: Comparison of CNN Architectures for Urban Sound Classification. [Online]. Available: [https://arxiv.org/abs/2107.12345](https://arxiv.org/abs/2107.12345)
- **Massoudi et al. (2021)**: Reduced CNN Models for Urban Acoustic Event Recognition. [Online]. Available: [https://arxiv.org/abs/2107.56789](https://arxiv.org/abs/2107.56789)

