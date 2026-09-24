# 🍕🥩🍣 Food Image Classification with PyTorch Transfer Learning

This project demonstrates **transfer learning for image classification using PyTorch and ResNet-50**.

A pretrained ResNet-50 model is adapted to classify food images into three categories:

* 🍕 Pizza
* 🥩 Steak
* 🍣 Sushi

Instead of training a convolutional neural network from scratch, the project uses a ResNet-50 model pretrained on ImageNet as a feature extractor and trains a new classification head for the target dataset.

---

## Project Overview

The goal of this notebook is to demonstrate how transfer learning can be used to build an image classifier with a relatively small dataset.

The workflow includes:

1. Loading a pretrained **ResNet-50** model.
2. Applying the preprocessing transforms associated with the pretrained weights.
3. Loading the pizza, steak, and sushi image dataset.
4. Freezing the pretrained feature-extraction layers.
5. Replacing the original ImageNet classifier with a new classifier for three classes.
6. Training the new classification layer.
7. Evaluating the model on test data.
8. Plotting training and validation loss/accuracy.
9. Making predictions on unseen images.

---

## Model Architecture

The model used in this project is:

**ResNet-50 + Transfer Learning**

The pretrained model is loaded using:

```python
weights = torchvision.models.ResNet50_Weights.DEFAULT

model = torchvision.models.resnet50(
    weights=weights
)
```

The convolutional feature-extraction layers are frozen:

```python
for param in model.parameters():
    param.requires_grad = False
```

The original classification layer is replaced with:

```python
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2),
    torch.nn.Linear(
        in_features=model.fc.in_features,
        out_features=3
    )
)
```

This reduces the number of trainable parameters substantially.

### Parameters

| Type                 | Parameters |
| -------------------- | ---------: |
| Total parameters     | 23,514,179 |
| Trainable parameters |      6,147 |
| Frozen parameters    | 23,508,032 |

Only the newly added classification head is trained.

---

## Dataset

The dataset contains images belonging to three food classes:

```text
pizza
steak
sushi
```

The data is automatically downloaded from the PyTorch Deep Learning course resources and organized into training and testing directories.

```text
data/
└── pizza_steak_sushi/
    ├── train/
    │   ├── pizza/
    │   ├── steak/
    │   └── sushi/
    │
    └── test/
        ├── pizza/
        ├── steak/
        └── sushi/
```

---

## Image Preprocessing

The project uses the preprocessing pipeline associated with the pretrained ResNet-50 weights:

```python
weights = torchvision.models.ResNet50_Weights.DEFAULT
auto_transforms = weights.transforms()
```

This ensures that the images are processed in the same way expected by the pretrained model.

The model receives images with dimensions:

```text
3 × 224 × 224
```

---

## Training Configuration

The model is trained using:

```text
Loss Function : CrossEntropyLoss
Optimizer     : Adam
Learning Rate : 0.001
Batch Size    : 32
Epochs        : 50
Random Seed   : 42
```

The notebook automatically uses CUDA when a compatible GPU is available:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

---

## Results

During training, the highest observed test accuracy was approximately:

**91.67%**

This occurred around **epoch 16**.

Example:

```text
Epoch: 16
Train Loss: 0.0438
Train Accuracy: 1.0000
Test Loss: 0.4864
Test Accuracy: 0.9167
```

Training accuracy reaches nearly 100% during several epochs, while test performance fluctuates later in training.

This suggests that further improvements could include:

* Early stopping
* Saving the best-performing model checkpoint
* Learning-rate scheduling
* Additional data augmentation
* Fine-tuning selected ResNet layers
* Experimenting with different learning rates

---

## Training Curves

The notebook visualizes both:

* Training and test loss
* Training and test accuracy

These curves can be used to inspect convergence and identify possible overfitting.

```python
plot_loss_curves(results)
```

---

## Making Predictions

A custom prediction function is implemented to classify new images.

```python
pred_and_plot_image(
    model=model,
    image_path=image_path,
    class_names=class_names
)
```

The function:

1. Loads the image.
2. Applies the required transformations.
3. Passes the image through the trained model.
4. Converts the output logits to probabilities using Softmax.
5. Selects the class with the highest probability.
6. Displays the image with its predicted label and probability.

The notebook demonstrates predictions on both test-set images and a custom pizza image.

---

## Technologies Used

```text
Python
PyTorch
TorchVision
ResNet-50
Transfer Learning
Matplotlib
Pillow
TorchInfo
Jupyter Notebook
CUDA
```

---

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd <repository-name>
```

Install the required packages:

```bash
pip install torch torchvision matplotlib torchinfo pillow requests jupyter
```

Then open the notebook:

```bash
jupyter notebook Pytorch_transfer_learning.ipynb
```

The dataset and supporting helper files are downloaded automatically when necessary.

---

## Environment

The notebook was executed using:

```text
PyTorch:     2.9.0+cu126
TorchVision: 0.24.0+cu126
```

The code is device-agnostic and can run on either CPU or CUDA-enabled GPU.

---

## Skills Demonstrated

This project demonstrates practical experience with:

**Deep Learning • Computer Vision • PyTorch • Transfer Learning • ResNet • Image Classification • Model Training • Model Evaluation • GPU Training • Inference**

---

## Future Improvements

Possible extensions to this project include adding more food categories, applying stronger data augmentation, fine-tuning deeper ResNet layers, implementing early stopping and model checkpointing, comparing ResNet-50 against architectures such as EfficientNet, and deploying the classifier as a web application or API.

---

## References

* PyTorch Transfer Learning for Computer Vision Tutorial
* TorchVision ResNet-50 Documentation
* PyTorch Deep Learning course resources by Daniel Bourke

---

## Author

**Mohamed Elshazli**

Machine Learning / AI Project
