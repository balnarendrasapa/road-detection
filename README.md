# Drivable Area and Lane segmentation

This project is about detecting the Drivable area and detecting lanes on the road. This project is mainly about finetuning the model with the dataset we generated using stable diffusion

## Dataset

- This dataset contains images for Drivable Area segmentation and Lane detection. All the images are generated using Stable diffusion in Google Colaboratory. This dataset is around 90 Megabytes. The project we are working on has two label outputs for each sample. And these outputs are overlayed on the original image.
- We've used stable diffusion to generate images for finetuning the model. click on the below badge to see how we worked with stable diffusion. The model we used is CompVis's stable-diffusion-v1-4 which can run on T4 GPU provided without any cost by google.

<a href="https://colab.research.google.com/github/balnarendrasapa/road-detection/blob/master/stable_diffusion/stable_diffusion.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Annotation

- The images are annotated using labelme tool. Which is an opensource tool used to annotate image data. Each image is annotated twice one is for drivable area segmentation and another is for lane detection.

### Labelme Annotation Tool

![image](https://github.com/balnarendrasapa/road-detection/assets/61614290/3458871a-12ff-4ce0-b26c-e0a57f985c96)

Click [here](https://github.com/wkentaro/labelme) to the labelme's github repo

#### Original Image

![image](https://github.com/balnarendrasapa/road-detection/assets/61614290/cda57ce3-14f0-4fec-aa8c-03974c25a753)

#### Annotation for Drivable area segmentation

![image](https://github.com/balnarendrasapa/road-detection/assets/61614290/c34f80fa-07e8-4b82-b767-9da4f8f14071)

#### Annotation for Lane detecton

![image](https://github.com/balnarendrasapa/road-detection/assets/61614290/d2ef6899-de98-41ea-a723-4498ae4454e6)

## Partitioning

The dataset is structured into three distinct partitions: Train, Test, and Validation. The Train split comprises 80% of the dataset, containing both the input images and their corresponding labels. Meanwhile, the Test and Validation splits each contain 10% of the data, with a similar structure, consisting of image data and label information.
Within each of these splits, there are three folders:

- Images: This folder contains the original images, serving as the raw input data for the task at hand.

- Segments: Here, you can access the labels specifically designed for Drivable Area Segmentation, crucial for understanding road structure and drivable areas.

- Lane: This folder contains labels dedicated to Lane Detection, assisting in identifying and marking lanes on the road.

## Accessing the Dataset

The dataset that we annotated is available in this repository in the datasets folder as datasets.zip. And also the dataset is pushed to huggingface datasets and you can access the dataset like shown below

### Python

<a href="https://colab.research.google.com/github/balnarendrasapa/road-detection/blob/master/datasets/Huggingface_dataset_tutorial.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

click on the above badge to see more on how to work with dataset

```python
from datasets import load_dataset

dataset = load_dataset("bnsapa/road-detection")
```
This dataset is hosted on huggingface. click [here](https://huggingface.co/datasets/bnsapa/road-detection) to the dataset card

### Through CLI

```bash
wget https://github.com/balnarendrasapa/road-detection/raw/master/datasets/dataset.zip
```

## Training

<a href="https://colab.research.google.com/github/balnarendrasapa/road-detection/blob/master/submissions/Update%202/Update_2_with_test.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

- Click on the above badge to open jupyter notebook that demonstrates how the model is finetuned.

### Transformation while Training

#### Random Perspective Transformation:
This transformation simulates changes in the camera's perspective, including rotation, scaling, shearing, and translation. It is applied with random parameters:

  - degrees: Random rotation in the range of -10 to 10 degrees.
  - translate: Random translation in the range of -0.1 to 0.1 times the image dimensions.
  - scale: Random scaling in the range of 0.9 to 1.1 times the original size.
  - shear: Random shearing in the range of -10 to 10 degrees.
  - perspective: A slight random perspective distortion.
  
#### HSV Color Augmentation:
  
  - This changes the hue, saturation, and value of the image.
  - Random gains for hue, saturation, and value are applied.
  - The hue is modified by rotating the color wheel.
  - The saturation and value are adjusted by multiplying with random factors.
  - This helps to make the model invariant to changes in lighting and color variations.
  
#### Image Resizing:

  - If the Images are not in the specified size, the images are resized to a fixed size (640x360) using cv2.resize.
  
#### Label Preprocessing:
  
  - The labels (segmentation masks) are thresholded to create binary masks. This means that pixel values are set to 0 or 255 based on a threshold (usually 1 in this case).
  - The binary masks are also inverted to create a binary mask for the background.
  - These binary masks are converted to PyTorch tensors for use in training the semantic segmentation model.

### Loss

- Tversky loss and Focal loss are used here. Total loss = Focal Loss + Tversky Loss

### Optimization

- In this setup, an Adam optimizer with a dynamically decreasing learning rate is employed. This adaptive learning rate is regulated using a Polynomial Learning Rate Scheduler, which gradually reduces the learning rate as the training progresses.

## Deployment

- The model is deployed on Huggingface spaces. click [here](https://huggingface.co/spaces/bnsapa/road-detection) to go there.
- You can deploy the model locally as well. There are three methods to run the model they are listed below.
  
#### Docker-Compose
- There is a docker image available with this repository. that is `road-detection`.
- git clone this repo. and `cd` into deployment and run `docker-compose up`.
- open `http://localhost:7860/` in you browser to see the app
  
#### Docker
- you can run the following command. This will download the image and deploy it. open `http://localhost:7860/` in you browser to see the app.
```bash
docker run -p 7860:7860 -e SHARE=True ghcr.io/balnarendrasapa/road-detection:latest
```

#### Python Virtual Environment
- `cd` into deployment directory. and run `python -m venv .venv` to create a virtual environment.
- run `pip install -r requirements.txt`
- run `python app.py`
- open `http://localhost:7860/` in you browser to see the app

## References

[1] [TwinLiteNet: An Efficient and Lightweight Model for Driveable Area and Lane Segmentation in Self-Driving Cars](https://arxiv.org/abs/2307.10705), **Authors**: Quang Huy Che, Dinh Phuc Nguyen, Minh Quan Pham, Duc Khai Lam, **Year**: 2023. Click [here](https://github.com/chequanghuy/TwinLiteNet) to go the TwinLiteNet Repository

[2] The Stable Diffusion code is taken from here. click [here](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb)
