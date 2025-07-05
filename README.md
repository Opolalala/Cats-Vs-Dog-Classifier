## Cats-Vs-Dog-Classifier
Cat vs Dog Classifier using CNN &amp; Data Augmentation
## 🐱🐶 Cat vs Dog Classifier using CNN & Data Augmentation

This project uses a Convolutional Neural Network (CNN) to classify images of cats and dogs. It is trained on the [TensorFlow Cats vs Dogs dataset](https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip), with additional data augmentation for better generalization.

---

## 📦 Dataset

The dataset is automatically downloaded from TensorFlow's public repo and contains two classes:
- `cats`
- `dogs`

Each is split into:
- `train/`
- `validation/`

---

## 🛠️ Key Features

- CNN model with 4 Conv2D + MaxPooling layers
- Data Augmentation (flip, rotation, translation, zoom)
- Binary classification (sigmoid output)
- Training and validation accuracy tracking
- Visual prediction on a single image

---

## 📊 Visualizations
You can visualize training loss/accuracy and prediction results using matplotlib.

## 📷 Sample Prediction
predict_image("your_image.jpg")

This will display the image and title the prediction as "Cat" or "Dog".

## 📌 Author
This project is part of a deep learning course using TensorFlow. All credits to the original course authors.
