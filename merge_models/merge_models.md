# CNN Model Fusion Strategies for Acoustic Event Recognition (AER)

In this section, we detail the different fusion strategies implemented for combining convolutional neural network (CNN) models aimed at improving performance in acoustic event recognition (AER). The directory is organized into two subfolders: one containing the models for transfer learning and reference models, and the other containing the similarity metric used to measure the degree of fusion between models. Additionally, we present four main fusion strategies.

## Fusion Strategies

- **Layer-wise Merge**: This strategy involves swapping entire layers between two models, evaluating the performance of each swap, and selecting the configuration that maximizes the F1-Score for both tasks.
- **Filter Merge Without Performance Evaluation**: Here, filters are merged by averaging their weights or selecting the maximum/minimum weight values without evaluating their performance during the merging process.
- **Performance-based Filter Merge Without Similarity Metric Evaluation**: This method swaps filters based on their performance on the test dataset, selecting filters that yield the highest average F1-Score.
- **Performance-based Filter Merge With Similarity Metric Evaluation**: This final strategy combines performance evaluation with a similarity metric, exchanging filters only if they surpass a similarity threshold.

## Subfolders

### Transfer Learning and Reference Models

This subfolder contains the specific models used for transfer learning and the reference models trained for two-class classifications, serving as benchmarks.

- **Base Model**: Trained on 12 classes from the UrbanSound8k dataset, achieving an F1-Score of 90%.
- **Specific Models**: These include models retrained for individual events like Gunshot, Siren, and Scream, each achieving F1-Scores of 97% or higher.
- **Two-Class Reference Models**: Models retrained to classify pairs of events, such as Gunshot-Scream, Gunshot-Siren, and Scream-Siren, with F1-Scores of up to 99%.

| Model             | F1-Score |
|-------------------|----------|
| Base Model        | 90%      |
| Gunshot           | 99%      |
| Siren             | 97%      |
| Scream            | 99%      |
| Gunshot - Scream  | 99%      |
| Gunshot - Siren   | 97%      |
| Scream - Siren    | 97%      |

### Similarity Metric

This subfolder includes the metric used to evaluate the similarity between models, facilitating the fusion process by identifying filters that can be exchanged or averaged.

- **Similarity by Filters Metric**: This metric is based on the Euclidean difference and norm of the filters, providing a similarity score between 0 and 1. A score close to 1 indicates high similarity.
  
![image](https://github.com/RicardoTangarife/AER_Sensor/assets/36963665/5c96abed-ae93-48a8-97cc-6d2cd0f4a1cf)
![image](https://github.com/RicardoTangarife/AER_Sensor/assets/36963665/c9e16e52-9627-431f-84dd-1ab8bafde29e)

### Example Results
Below are example results of the similarity metric applied between two models, demonstrating how similarity values help in the fusion process.

Similarity Metric Results for Models Trained by Transfer Learning Comparison of Gunshot vs. Screams with Normalized Filters, Configuration 1.
![image](https://github.com/RicardoTangarife/AER_Sensor/assets/36963665/587eefdd-4d6a-42f3-bbdd-963da8755853)


### Comparative Summary of Fusion Strategies
Finally, we present a comparative summary of the performance of the merging strategies applied to the different acoustic event models studied. The most effective merging strategies identified were filter merging evaluating performance without similarity metrics, and filter merging with similarity metrics using a threshold below 80%, both optimized towards the average. These strategies outperformed the average performance of individual models and the reference two-class model. Among these alternatives, filter merging without similarity metrics stands out as the computationally less expensive option.
Additionally, the filter merging strategy with average optimization achieved good performance with the different merged models and is computationally less costly than the performance-based strategies, as it does not require evaluating the model on the test data when performing merge for each filter.

![image](https://github.com/RicardoTangarife/AER_Sensor/assets/36963665/42fd3b71-2d34-4015-a303-74fbd81de1d2)


