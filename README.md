# Drivable Area and Lane segmentation

This project is about detecting the Drivable area and detecting lanes on the road. This project is mainly about finetuning the model with the dataset we generated using stable diffusion

## Dataset

- This dataset contains images for Drivable Area segmentation and Lane detection. All the images are generated using Stable diffusion in Google Colaboratory. This dataset is around 90 Megabytes. The project we are working on has two label outputs for each sample. And these outputs are overlayed on the original image.

## Annotation

- The images are annotated using labelme tool. Which is an opensource tool used to annotate image data. Each image is annotated twice one is for drivable area segmentation and another is for lane detection.

## Partitioning

The dataset is structured into three distinct partitions: Train, Test, and Validation. The Train split comprises 80% of the dataset, containing both the input images and their corresponding labels. Meanwhile, the Test and Validation splits each contain 10% of the data, with a similar structure, consisting of image data and label information.
Within each of these splits, there are three folders:

- Images: This folder contains the original images, serving as the raw input data for the task at hand.

- Segments: Here, you can access the labels specifically designed for Drivable Area Segmentation, crucial for understanding road structure and drivable areas.

- Lane: This folder contains labels dedicated to Lane Detection, assisting in identifying and marking lanes on the road.

## Accessing the Dataset

The dataset that we annotated is available in this repository in the datasets folder as datasets.zip. And also the dataset is pushed to huggingface datasets and you can access the dataset like shown below

### Python

```python
from datasets import load_dataset

dataset = load_dataset("bnsapa/road-detection")
```

### Through CLI

```bash
wget https://github.com/balnarendrasapa/road-detection/raw/master/datasets/dataset.zip
```

## References

[1] [TwinLiteNet: An Efficient and Lightweight Model for Driveable Area and Lane Segmentation in Self-Driving Cars](https://arxiv.org/abs/2307.10705), **Authors**: Quang Huy Che, Dinh Phuc Nguyen, Minh Quan Pham, Duc Khai Lam, **Year**: 2023. Click [here](https://github.com/chequanghuy/TwinLiteNet) to go the TwinLiteNet Repository
