# Pascal VOC 2008 Classification Challenge

This project aims to classify images from the Pascal VOC 2008 dataset using various deep learning architectures. The task involves training models on the provided **train** and **val** datasets and evaluating performance using **Mean Average Precision (mAP)**.

## Group Members

- **Group Leader**: [Muhammad Sameed]
- **Group Members**: [Muhammad Sameed], [Ayesha Irshad], [Arooba Minhas], [Abdullah Baqai]

## Problem Understanding

The **Pascal VOC 2008** dataset consists of annotated images for various object detection and classification tasks. For this project, we focus on the **classification** task, where images are labeled with one or more object categories. The goal is to classify images into these categories based on the training data and evaluate the model's performance on the validation set.

## Data

- **Training Data**: Used to train the models.
- **Validation Data**: Used to evaluate model performance.
- The **test data** is not used directly in this challenge, as we work with the validation set for testing purposes.

The data is available at the [Pascal VOC 2008 site](http://host.robots.ox.ac.uk/pascal/VOC/voc2008/index.html).

## Evaluation Criteria

The performance is measured using **Mean Average Precision (mAP)**, which calculates the precision at different recall levels. This evaluation method is suitable for multi-label classification tasks where each image can belong to multiple classes.

## Approach

### 1. **Transfer Learning with VGGNet**

We begin by applying **VGGNet** through transfer learning for the classification task. We compare the results with three other CNN architectures:
- **VGGNet**
- **ResNet50**
- **DenseNet**
- [Additional Architecture]

For each architecture, we evaluate the performance based on **mAP** and provide a comparison between models. We also display the **top 10 ranked images** for each architecture.

### 2. **Mean Average Precision (mAP)**

We explain how **mAP** works through examples, showing how it measures the quality of classification across multiple classes in the dataset.

### 3. **Variational Autoencoder (VAE) for Data Augmentation**

We generate **random data** for each class using a **Variational Autoencoder (VAE)** trained on the training samples only. We then repeat the classification process using the augmented data and plot the **mAP** vs. the number of generated samples (100, 200, 500) per class.

#### Discussion

We discuss whether the augmentation improved the model's performance and the potential reasons behind it.

### 4. **Generative Adversarial Network (GAN) for Data Augmentation**

We use a **Generative Adversarial Network (GAN)** to generate synthetic data for each class, again using only the training samples. The models are trained and evaluated on the augmented dataset, and the results are compared to the original data. We plot the **mAP** vs. the number of generated samples (100, 200, 500).

#### Discussion

We analyze whether GAN-generated data improves classification performance and the factors influencing the results.

## Results
Model	 	| mAP 	   | Accuracy | Precision | Recall
ResNet50 	| 0.703557 | 0.937486 | 0.534986  | 0.801817
VGG16 	 	| 0.672096 | 0.967301 | 0.886091  | 0.610239
DenseNet121 	| 0.78855  | 0.97301  | 0.829636  | 0.772089
MobileNet_V2 	| 0.692244 | 0.967647 | 0.831633  | 0.672998


### Augmentation Performance
- **VAE Augmentation**:
  - 100 samples/class: [mAP Value]
  - 200 samples/class: [mAP Value]
  - 500 samples/class: [mAP Value]
- **GAN Augmentation**:
| Model        |      mAP |   Accuracy |   Precision |   Recall |
|:-------------|---------:|-----------:|------------:|---------:|
| VGG16        | 0.941649 |   0.986974 |    0.992982 | 0.744737 |
| RESNET50     | 0.949177 |   0.863421 |    0.267983 | 1        |
| DENSENET121  | 0.95     |   1        |    1        | 1        |
| MOBILENET_V2 | 0.94929  |   0.999737 |    1        | 0.994737 |

## Conclusion

This project demonstrates how **transfer learning** can be applied to the **Pascal VOC 2008** classification challenge, and it explores the impact of data augmentation techniques like **VAE** and **GAN**. Through the evaluation of **mAP**, we gain insights into the effectiveness of various approaches in improving model performance.

## References

- Pascal VOC 2008 Dataset: [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2008/index.html)
- [VGGNet](https://arxiv.org/abs/1409.1556)
- [ResNet50](https://arxiv.org/abs/1512.03385)
- [DenseNet](https://arxiv.org/abs/1608.06993)
- [Variational Autoencoders](https://arxiv.org/abs/1312.6114)
- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
