# NanoChest-net
NanoChest-net CNN for Radological Studies Classification

## Dataset
We have used the following datasets:
- NIH Tuberculosis dataset.
- CELL pneumonia dataset v3.
- The COVID-19 Data Image Collection.
- RSNA Pneumonia Challenge.
- BCDR Breast Cancer datasets.


Dataset links (in orden of appearance):
- https://lhncbc.nlm.nih.gov/LHC-publications/pubs/TuberculosisChestXrayImageDataSets.html 
- https://data.mendeley.com/datasets/rscbjbr9sj/3
- https://github.com/ieee8023/covid-chestxray-dataset
- https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/ 
- https://bcdr.eu/

Please refer to original papers for more details.

### Architecture
We have proposed a new efficient an minimalist CNN architecture to classify radiological studies called NanoChest-net.

Attention! We have encountered that our table and diagram from the paper is incorrect. An extra Conv2D and relu, on the input of the network has been found. Code from this repository is correct according with results presented on the paper.

We have compared our results with baseline CNN models such as ResNet50, Xception, DenseNet121.

### Pretrained models
Pretrained models can be found in Keras repository: https://keras.io/api/applications/

# Results
We have published a paper that will be published within 5 days from now (April 23rd, 2021).

Luján-García, J.E.; Villuendas-Rey, Y.; López-Yáñez, I.; Camacho-Nieto, O.; Yáñez-Márquez, C. NanoChest-Net: A Simple Convolutional Network for Radiological Studies Classification. Diagnostics 2021, 11, 775. https://doi.org/10.3390/diagnostics11050775

# How to run
We have implemented our experiments using TensorFlow 2.1.0 with Keras as high-level DL framework; sci-kit learn 0.23.2; and OpenCV 3.4.2.

If you want to run your own experiments, please clone this repository. Please try the following steps:
- Clone full repository.
- Edit utilities.py in order to change your own path in which your dataset is contained.
- You need a structure for you dataset as follows:
- dataset_folder/
  - traning/
    - class1/
    - class2/
  - validation/
    - class1/
    - class2/
  - testing/
    - class1/
    - class2/  
- You will need to create a csv file "path-to-your-model/Experiments_info.csv" to save the results.
- Run NanoChest-net_exp.py editing your own preferences such as number of dataset, lr, optimizer, etc.

