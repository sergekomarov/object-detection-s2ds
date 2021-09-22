# S2DS-Aug2021-CV

The project aims to develop an object detection system to identify two types of controllers, Siemens S7-300 and Allen-Bradley (CompactLogix) L24ER-QB1B, in PLC cabinets.

## Prerequisites when running locally

1. Tensorflow
2. Tensorflow Object Detection API

A walk-through on how to install these two is provided in https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html. This tutorial also shows how to install GPU support.

## Assumed project structure:

```
main_project_dir/
├─ README.md                               
├─ generate-synthetic-images/              <- scripts to generate synthetic images
├─ data/                  
│   ├─ test-annotated-images/              <- test images with xml annotations
│   ├─ synthetic-train-annotated-images/   <- synthetic images with xml annotations used for training
│   ├─ test-google-images/                 <- a handful of images of other cabinets from Google search that we can use for final testing
│   ├─ train-valid-split
│   │    ├─ training/                      <- preprocessed training images with xml annotations
│   │    ├─ validation/                    <- preprocessed validation images with xml annotations
│   │    ├─ train.record                   <- TF .record file containing the training data
│   │    ├─ valid.record                   <- TF .record file containing the validation data
│   │    └─ label_map.pbtxt                <- label map file that maps class IDs to class names
├─ models/                              
│   ├─ pre-trained/                        <- pre-trained models downloaded from TF Object Detection Model Zoo
│   ├─ fine-tuned/                         <- fine-tuned models trained on the data in train-valid-split
├─ notebooks/                              <- notebooks used for training and evaluation
│   ├─ training.ipynb
│   ├─ training_validation.ipynb
│   └─ evaluation.ipynb
└─ src/                                    <- modules used in the notebooks
```

## Labelling images

In order to train a model, the training and validation images need to be annotated first. Because in our project, we only use synthetic data for training, the training images are annotated automatically as part of the image synthesis process. The validation/evaluation images however need to be annotated manually.

A convenient way to do so is by using `labelImg`. It can be installed by running `pip install labelImg`. A tutorial is given here: https://medium.com/deepquestai/object-detection-training-preparing-your-custom-dataset-6248679f0d1d. For each image, `LabelImg` generates an .xml annotation file with the same name as the original image. In our project, we use the PASCAL VOC format of annotation. These .xml files contain the bounding boxes and classes of the objects present in the images.

## Training a model from Tensorflow Detection Model Zoo

The training.ipynb notebook shows how to train an object detection model on synthetic images using transfer learning.

This includes how to:
* Preprocess the training data, including data augmentation
* Form training and validation datasets
* Convert image data with annotations into TF .record files (the input format expected by TF models)
* Obtain and configure a pre-trained model from TF Model Zoo
* Train / fine-tune the model
* Track training and validation performance metrics

## Evaluating a model on a set of test images

The evaluation.ipynb notebook demonstrates evaluation of a given exported model.

This includes how to:
* Load a model
* Calculate the COCO metrics for test predictions
* Visualize metrics and confusion matrices
* Perform visual assessment of predictions
* Do inference on images of other cabinets to see if the model generalizes well

## Environment

The notebooks can be either run locally from their folder or on Google Colab. When run on Colab, the notebooks include the necessary initialization, including the installation of TF Object Detection API. The training notebook also allows one to employ the Colab TPU optionally.

## Authors

* Agnieszka Czeszumska (agaczesz@gmail.com)
* Frederic Colomer Martinez (frederic.colomer@gmail.com)
* Jenny Shih (jennywho86@gmail.com)
* Lennart Schmidt (lennartschmidt90@gmail.com)
* Sergey Komarov (skomarov1000@gmail.com)

### Installing requirements
------------

    pip install -r requirements.txt

Note that errors might arise for numpy versions >1.17.4.
