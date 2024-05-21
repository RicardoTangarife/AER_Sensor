## Dataset

For the selection of events of interest, our focus is on public safety, identifying crucial acoustic events such as sirens, gunshots, and screams for early detection of hazardous situations. These distinctive sounds can help prevent incidents and facilitate effective responses. Our aim is to provide effective tools for identifying threats in urban environments, contributing to the protection and well-being of the community. These events are part of a larger project serving as a case study in the field of public safety.

### Dataset Description

The training of our Acoustic Event Recognition (AER) models was based on the use of relevant datasets suitable for the task. Primarily, we employed the UrbanSound8k dataset, which organizes a variety of urban sounds into ten categories, including:

- Air conditioning
- Car horn
- Children playing
- Drill
- Dog barking
- Engine
- Gunshot
- Jackhammer
- Siren
- Street music

This dataset consists of 8732 audio files, each lasting up to 4 seconds, recorded at a sampling frequency of 22.05 KHz.

### Siren Recognition Model

For training the AER model dedicated to siren recognition, we balanced the number of samples related to the siren class from the UrbanSound8k dataset. The other nine categories were labeled as non-interest events. The model was trained with:

- 929 samples for siren events
- 929 samples for non-interest events

### Gunshot Recognition Model

To enhance the robustness of the gunshot recognition model, we enriched the UrbanSound8k dataset with explosion sounds from the Sound Events for Surveillance Applications (SESA) dataset. These additional sounds were normalized to a maximum duration of 4 seconds. The balanced training dataset included:

- 374 samples for gunshot events
- 374 samples for non-interest events

### Scream Recognition Model

For the scream recognition model, we utilized a dataset of human screams available on the Kaggle platform. After cleaning and preprocessing, each sample was standardized to a maximum duration of 4 seconds. The balanced training dataset consisted of:

- 391 samples for scream events
- 391 samples for non-interest events

Each of these models was developed following the same approach of balancing and random selection of non-interest events from the UrbanSound8k dataset, ensuring a robust and comprehensive training process.


| Acoustic Event | Database                                             | Classes                                                                   | Samples Number |
|:--------------:|:---------------------------------------------------:|:-------------------------------------------------------------------------:|:--------------:|
|    Gunshot     |               UrbanSound8k                          |                                  Gunshot                                  |      374       |
|  Non-Gunshot   |               UrbanSound8k                          | Air conditioning, car horn, children playing, dog barking, drill, engine, jackhammer, siren, and street music |      374       |
|                | Sound Events for Surveillance Applications (SESA)   |                                Explosion                                 |                |
|     Siren      |               UrbanSound8k                          |                                  Siren                                    |      929       |
|  Non-Siren     |               UrbanSound8k                          | Air conditioning, car horn, children playing, dog barking, drill, engine, jackhammer, gunshot, and street music |      929       |
|     Scream     |           Screams Dataset Kaggle                    |                                 Screams                                   |      391       |
|  Non-Scream    |               UrbanSound8k                          | Air conditioning, car horn, children playing, dog barking, drill, engine, jackhammer, gunshot, siren, and street music |      391       |
