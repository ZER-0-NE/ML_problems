# ML_problems

This is a repository which contains my Machine learning problems and scripts created while learning various implementations in ML.

## Usage

You can run all the scripts individually by having all the required dependencies installed in your machine. Please note that you should have all the other scripts in your same local directory in order for the imports to work properly.

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


### Approach used:
 We use the **ResNet-50 architecture** which is based on residual learning network, easier to optimize and consequently, enables training of deeper networks.
This leads to an overall improvement in network capacity and performance.
It **won the ILSVRC 2015 classification task** by obtaining a 28% relative improvement on the COCO object detection dataset.


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## License
[MIT](https://choosealicense.com/licenses/mit/)