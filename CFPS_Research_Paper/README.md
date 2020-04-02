## Detection of Rare Genetic Diseases using Facial 2D Images with Transfer Learning

### Overview:
With a population of around 7.6 billion in the world, approx 61 crore people in the world suffer from rare genetic diseases. This is a massive report and we still don’t see proper care given to them. They are an ignored part of our community and we need suitable actions to be taken in favor of them. Though we cannot cure most of them completely, we can surely help improve their lives with our little gestures.

![Rare Diseases](/assets/rare_dis.jpg)

From the 60 crore people affected from rare genetic diseases, **India has around 10% of them**.

### Difficulties for a Doctor:
There are a number of reasons why this statistic goes unnoticed. Of all the doctors, only a limited number of individuals have the proper training and ability to recognize these disorders. They have to depend on the facial features which occur in 30-40% of the individuals. One of the reasons of why we use facial images in this research.

### The 12 Classes:
- 22q11.2DS                              
- Angelman
- Apert
- Cornelia de Lange (CDL)
- Down’s Syndrome
- Hutchinson-Grifford Progeria
- Marfan
- FragileX
- Sotos
- Treacher Collins
- Turner
- Williams 


### Dataset:
The eLife dataset was provided by [Cristoffer Nellaker](https://elifesciences.org/articles/02020) from Oxford University (excluding the Gorlin Collection).

### Approach used:
 We use the **ResNet-50 architecture** which is based on residual learning network, and is easier to optimize and consequently, enables training of deeper networks.
This leads to an overall improvement in network capacity and performance.
It **won the ILSVRC 2015 classification task** by obtaining a 28% relative improvement on the COCO object detection dataset.

![ResNet50](/assets/res50.jpg)

### Methodology:
First, we used 4 Fully Connected layers on top of VGGFace.
Then, 3 fully connected layers and an SVM classifier on top.
The Transfer Learning approach allows us to use already learned features again.

<p align="center">
  <img  src=/assets/methods.png/>
</p>

- SGD optimizer with very small learning rate of 0.0001 and momentum of 0.9 was used.
- Callback was defined in keras for reducing learning rate by 0.1 whenever the validation loss stopped improving.
- Batch size was set at 32 in each case.
- We minimize cross-entropy loss as a basis of good classification.
- The classifier was built in keras (v2) library for deep learning.
- The model was trained using Tesla K80 GPU.
- Total of 200 epochs were run and the accuracies stabilised around 100 epochs.
- We calulated prediction confidence for each image belonging to the 12 classes.

### Comparison:

We compare our model with that of the state-of-the-art. The below tables show our comparison.


<p align="center">
  <img  src=/assets/comparison.png/>
</p>


#### Confusion Matrix:


<p align="center">
  <img  src=/assets/conf_matrix.png/>
</p>



<p align="center">
  <img  src=/assets/conf_matrix_.png/>
</p>



### Accuracy Results:


<p align="center">
  <img  src=/assets/train.png/>
</p>


<p align="center">
  <img  src=/assets/train_.png>
</p>
