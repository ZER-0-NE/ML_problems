# ML-DL-Practice

This is a repo I use often to brush up my skills in Tensorflow/Keras and to learn and keep track of some minute things which often go unnoticed.

## INDEX

- Batch Norm
  - - [x] Theoretical background
  - - [ ] Scratch implementation
  - - [x] Code sample(s)


- Callbacks
  - - [x] Theoretical background
  - - [ ] Scratch implementation
  - - [x] Code sample(s)


- Gradient Descent
  - - [ ] Theoretical background
  - - [ ] Scratch implementation
  - - [ ] Code sample(s)


- Metrics in Keras
  - - [x] Theoretical background
  - - [x] Scratch implementation
  - - [x] Code sample(s)

- Regularizers
  - - [x] Theoretical background
  - - [ ] Scratch implementation
  - - [ ] Code sample(s)

- Supervised Learning
  - - [x] Decision Tree (C/R)
  - - [x] Random Forest (C/R)
  - - [ ] Support Vector Machines (# Error)
  - - [ ] kNN (C/R)
  - - [ ] Naive Bayes

- Unsupervised Learning
  - - [x] K means clustering
  - - [ ] Association Rules 

## Usage

You can run all the scripts/notebooks individually by having all the required dependencies installed in your machine. Please note that you should have all the other scripts in your same local directory in order for the imports to work properly.

## Calculating the number of parameters in Feed-forward Neural Networks:

Let, 
- i, input size
- h, size of hidden layer
- o, output size

For one hidden layer, num_params
= connections between layers + biases in every layer

`num_params = (i×h + h×o) + (h+o)`

#### Example 1:


<p align="center">
  <img  src=/assets/param_1.png/>
</p>


Here, i = 3; h = 5; o = 2

num_params = connections between layers + biases in every layer

= (3×5 + 5×2) + (5+2)

= 32

Building such a sequential model would look something like:

```
model = Sequential([
	Dense(5, input_shape = (None, 3))
	Dense(2)
])
```

#### Example 2:


<p align="center">
  <img  src=/assets/param_2.png/>
</p>


Here, i = 50; h = 100, 1, 100; o = 50

num_params = connections between layers + biases in every layer

= (50x100 + 100x1 + 1x100 + 100x50) + (100+1+100+50)

= 10451

Building such a sequential model would look something like:

```
model = Sequential([
	Dense(100, input_shape = (None, 50)),
	Dense(1),
	Dense(100),
	Dense(50),
])
```

## Calculating the number of parameters in Convolutional Neural Networks:

Let, 
- i, no. of input maps (or channels)
- f, filter size/dimensions of the filter
- o, no. of output maps

For one conv layer, num_params
= weights + biases

`num_params = [i x (fxf) x o] + o`

#### Example 1:


<p align="center">
  <img  src=/assets/param_3.png/>
</p>


Here, i = 1 (only one channel); f = 2; o = 3

num_params = weights + biases

= [1 x (2x2) x 3] + 3

= 15 (There are 15 parameters viz., 12 weights and 3 biases)

Building such a sequential model would look something like:

```
model = Sequential([
	Conv2D(3, (2,2), input_shape = (None, None, 1))
])
```

#### Example 2:


<p align="center">
  <img  src=/assets/param_4.png/>
</p>


Here, i = 3 (three channels); f = 2; o = 1

num_params = weights + biases

= [3 x (2x2) x 1] + 1

= 13 (There are 13 parameters viz., 12 weights and 1 bias)

Building such a sequential model would look something like:

```
model = Sequential([
	Conv2D(1, (2,2), input_shape = (None, None, 3))
])
```

[REFERENCES](https://towardsdatascience.com/counting-no-of-parameters-in-deep-learning-models-by-hand-8f1716241889#192e)

## Weights and Biases

### Default Weights and Biases

Generally, we do not specify the weights and biases in the models we use.
The default values of weights and biases in TensorFlow depend on the type of layers we are using.
For example, in a *Dense* Layer, the biases are set to zero by default and the weights are set according to the `glorot_uniform`, the *Glorot Uniform* Initializer.

The Glorot uniform initializer draws the weights uniformly at random from the closed interval [-c, c], where

<p align="center">
  <img  src="https://bit.ly/2UL6S6H"/>
</p>

and n <sub>input</sub> and n <sub>output</sub> are the number of inputs to and outputs from the layer respectively.

### Initialize your own Weights and Biases

TensorFlow makes it easier to initialize your own weights and biases. When defining a model, we can use `kernel_initializer` and `bias_initializer` to set the weights and biases respectively.

For example.

```
model = Sequential([
     Conv2d(64, (3,3), input_shape = (128, 128, 1), kernel_initializer = 'random_uniform', bias_initializer = 'zeros', activation = "relu"),
     MaxPooling1D(pool_size = 4),
     Flatten(),
     Dense(64, kernel_initializer = 'he_uniform', bias_initializer = 'ones', activation = "relu")
])
```

We can also initialize in the following way:

```
model.add(Dense(64, 
                kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05), 
                bias_initializer=tf.keras.initializers.Constant(value=0.4), 
                activation='relu'),)

model.add(Dense(8, 
                kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0, seed=None), 
                bias_initializer=tf.keras.initializers.Constant(value=0.4), 
                activation='relu'))
```

### Custom Weights and Bias Initializer

It is also possible to define your own weights and bias initilizers. Initializers must take in two arguments, the `shape` of the tensor to be initilised and its `dtype`.

For example,

```
import tensorflow.keras.backend as K

def my_init(shape, dtype=None):
	return K.random_normal(shape, dtype=dtype)

model.add(Dense(64, kernel_initializer=my_init))
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## License
[MIT](https://choosealicense.com/licenses/mit/)
