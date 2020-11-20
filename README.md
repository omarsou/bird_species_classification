# bird_species_classification
private kaggle challenge : https://www.kaggle.com/c/mva-recvis-2020/

# Bird Species Classification

## 1. Introduction

The task involves a supervised classification of bird species
from ​different classes adapted from the ​CUB-200-2011 dataset​.
The goal was to produce a model that can give the highest
accuracy on the test set containing the same categories.

## 2. Preprocessing

### 2.1 Size and kind of interpolation
In the “resize_interpolation_analysis” notebook, I tried for one
fixed size image different interpolation technique. I took the one
that gave me the best validation accuracy (Bicubic). I did the
same thing for the size of the image, and I took the one that gave
me the best validation accuracy (334x334)

### 2.2 Birds Image Retrieval
Next, as I noticed that the background was not a discriminating
factor, I wanted to remove it. In the “crop_image_generation”
notebook, thanks to the pretrained model MASK RCNN
ResNet50, I was able to extract more than 85% of the birds. Then
I created a second new dataset with cropped images (training,
validation & test). As the shape of the cropped images was
between 150/250, I chose to resize them to 224x224 using the
LANCZOS interpolation.

## 3. Transfer Learning

I have used Transfer Learning but with different
approaches that give me different results. I will detail them
in the following paragraphs. I used Keras as a deep
learning framework.

### 3.1 InceptionResnetV2 / InceptionV3 / Xception
This approach is mainly based on the following paper “​ **Bird
Species Classification using Transfer Learning with
Multistage Training** ​”. I used InceptionResnetV2, InceptionV
and Xception as the base model architecture. Concerning the top
layer, I used a simple AveragePooling2D => Flatten => Dense
=> Dropout => Dense. I trained each model on both the original
dataset and the new dataset composed of cropped images with
the same optimizer (Adam). As the last layers of the
pretrained-model are the one that are specific to the tasks they
were pre-trained on, I fine-tuned each base model by unfreezing
a certain number of his last layers (I select the number of
unfreezed layers that give the best accuracy and F1-score on the
validation set). At a certain number, the model overfitted on the
train set. The results on the train & validation set can be found
in table 1. I went with a Bagging Ensemble Method with some
particularities, I train all the models on the same data.:
=> Max Probability: Returns the label of the model that give
the highest probability : ​ **0.80645** ​public score
=> Voting Method : ​ **0.81290** ​public score
Notebooks : [imagenet_ensemble_models,
inception_resnet_v2_finetune, inception_v3_finetune,
xception_finetune]

### 3.2 INaturalist Pre-Trained Model
I downloaded and adapted the pre-trained model from this link :
https://github.com/richardaecn/cvpr18-inaturalist-tran . I did the same finetuning than before but only on
InceptionV3. At the end, I had two models : InceptionV3 trained
on original images and the one trained on the cropped images.
=> Returns the label of the model that give the highest
probability : ​ 0.87741 ​public score
=> Same thing but if the difference between the two probability
is less than 0.15, I will take the prediction on the original image
(as the inceptionV3 on original performs better) : ​ 0.88387 ​public
score.
Notebooks : [inaturalis_ensemble_model]

## 4. Transfer Learning + Attention

The goal here is to use the attention mechanism to help the
model focus on the regions of the picture that are the most
relevant. For our case, it’s the birds. So It’s like replacing what
we have done before with our cropped images. I kept the
cropped images because there are probably some bird’s body
regions that can be more relevant than the others.
### 4.1 Global Weighted Average Pooling
The implementation is based on this kaggle notebook https://www.kaggle.com/kmader/gpu-pretrained-vgg16-hratten-tuberculosis
 It turns pixels in the GAP on and off before the pooling and
rescale the results based on the number of pixels. For each
image, he built an attention map.
As a base model, I used InceptionV3 and InceptionResNetV2,
each one of them was combined with the attention block, and
trained on both the original images and the cropped images. I had
4 models at the end, I added two models for “regularization” (the
InceptionV3 models trained during 3.1). The method chosen here
is Stacking Models based on the kdnuggets articlehttps://www.kdnuggets.com/2017/02/stacking-models-imropved-predictions.html​.
At the end, I had two models, one for the original image and
another one for the cropped images. I used the max probability
approach described on 3.1 : ​ 0.84516 ​on the public score.
Notebooks : [stacking_models, attention_models]
### 4.2 Cascade Attention
The implementation is based on this paper ​https://www.researchgate.net/publication/336735157_Learning_Cascade_Attention_for_fine-grained_image_classification along with their implementation on github ​https://github.com/billzyx/LCA-CNN. To sum-up, we have
two parallel pretrained-model as the base architecture. It presents
a global attention pooling that receives the attention binary mask
from the first pretrained-model (InceptionV3 - 1), then uses it as
a filter to make the second pretrained-model (InceptionV3 - 2) to
pay attention to the selected region. It also has a Global Average
Pooling that will take only the features from the first
pretrained-model. Then they cross the two outputs from the
global attention pooling and the global average pooling.
Concerning the ensemble learning, the approach is based on the
Random Forest Algorithm. I randomly split the train set into
three subsets (These subsets are not disjointed!). At the end, I
had three models, each one of them trained on one of the subset.
Concerning the predictions, I used a simple Voting Classification
: ​ 0.70967 ​on public score.
As far as I am concerned, these models have overfitted on the
train set even if the validation accuracy was good (which
probably due to the fact that the validation images were “easy” to
classify compared to the test images).
I should have used the cropped images, unfortunately I was
running out of GPU (both Kaggle & Colab).
Notebooks : [cascade_attention]

## 5. Conclusion

I have compiled all the results in Table 2.
I really liked this small project, it was not complicated but it
urged me to do a lot of research on what is currently being done
on Image Classification.
I can see that my approach was unstructured and not rigorous.
I will pay more attention to that for my future projects.


## TABLE 2
Approach | Score |
--- | --- |
Approach 3.1 | **0.81290** | 
Approach 3.2 | **0.88387** | 
Approach 4.1 | **0.84516** |
Approach 4.1 | **0.70967** |  
