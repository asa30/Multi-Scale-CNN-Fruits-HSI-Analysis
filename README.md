# Investigating the Performance of Multi-Scale/Kernel CNNs with Attention at Analyzing Hyperspectral Images

This readme file will serve as the walkthrough to navigating this project. 

all_models.py is a file that has all models used in this project. 

hsi_datagen.py is the file containing a custom data generator suited for this project. 

.env and dmog.env are files containing the working and data directories paths respective of the development environment. 

There are many directories containing checkpoints/weights of each model. 

The directories ending with "_checkpoints" contains the checpoints of models trained in the jupyternotebooks, 

whilst the directories ending with "_loocv" contain the models weights at the end of each iteration in an LOOCV approach.

## Expirementations as well as replicating literature results

Files Involved: miscellaneous/attempt2.ipynb, main.ipynb

These two notebooks contain all my trials and errors and my steps taken to develop the expiremental architectures. 

miscellaneous/attempt2.ipynb is the notebook that generated all most of the checkpoints of the HSCNN model which were later used to create the confusion matrices. 

It also contains expiremental archiectures that were attempted, trained and tested using a conventional train-test split approach which was later abandoned due to its unreliability. 

The confusion matrices associated with the HSCNN model are to be found generated in Confusion_Matrices.ipynb

## GAN Expirement 

Dirctory: GAN 

This directory contains the GAN model implemented as well as its architecture (in h5) for each attempt.

## Simple ML algorithms

* DT
* RF
* KNN
* SVM

for each of these directories many python scripts will be found, scripts that implement the model using the conventional train-test split (to compare with the replicated results of HSCNN) and LOOCV for each subset of data (Kiwi, Avocado x VIS,NIR). 

In addition to these python files, there are bash scripts that were necessary to deploy the python scripts as batch jobs on dmog. The output and error files generated by these batch jobs are also available.

## Learning Curves

Directory: TRAIN

To generate the Learning curves for each model I trained them on the entire dataset to observe how they learn from hyperspectral images. given the nature of the computing environment I had parsed the output files and generated the learning curves in Learning_Curves_Visualisations.ipynb for ease of access.

## LOOCV Results 

Directories involved: 

* LOOCV
* DT
* RF
* KNN
* SVM

The LOOCV Results for the simple ML algoritms can be found in their respective directories, whilst those for the DL models will be found in LOOCV within their respective subdirectories.

## How to replicate my results 

If they are to be run on dmog then it can be simply run by using the sbatch command with the bash files as arguments, it must to be noted that an exception to this is a jupyter notebook which was run on my personal computer. 

The dataset can be acquired [here](https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/)
