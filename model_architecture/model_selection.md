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

- **Chen et al. (2017)**: Chen, Y., Zhang, Y., and Duan, Z. (2017). Dcase2017 sound event detection using convolutional neural network. Detection and Classification of Acoustic Scenes and Events. Available at: [https://dcase.community/documents/challenge2017/technical_reports/DCASE2017_Chen_124.pdf](https://dcase.community/documents/challenge2017/technical_reports/DCASE2017_Chen_124.pdf)
- **Doshi et al. (2022)**: Doshi, S., Patidar, T., Gautam, S., and Kumar, R. (2022). Acoustic Scene Analysis and Classification Using Densenet Convolutional Neural Network. Tech. rep., EasyChair. Available at: [https://www.techrxiv.org/doi/pdf/10.36227/techrxiv.19728895.v1](https://www.techrxiv.org/doi/pdf/10.36227/techrxiv.19728895.v1)
- **Tsalera et al. (2021)**: Tsalera, E., Papadakis, A., and Samarakou, M. (2021). Comparison of pre-trained cnns for audio classification using transfer learning. Journal of Sensor and Actuator Networks 10, 72. Available at: [https://www.mdpi.com/2224-2708/10/4/72](https://www.mdpi.com/2224-2708/10/4/72)
- **Massoudi et al. (2021)**: Massoudi, M., Verma, S., and Jain, R. (2021). Urban sound classification using cnn. In 2021 6th International Conference on Inventive Computation Technologies (ICICT) (IEEE), 583–589. Available at: [https://www.researchgate.net/profile/Massoud-Massoudi/publication/349660725_Urban_Sound_Classification_using_CNN/links/615fd85b1eb5da761e5e0dd5/Urban-Sound-Classification-using-CNN.pdf](https://www.researchgate.net/profile/Massoud-Massoudi/publication/349660725_Urban_Sound_Classification_using_CNN/links/615fd85b1eb5da761e5e0dd5/Urban-Sound-Classification-using-CNN.pdf)

