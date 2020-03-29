# Tensorflow-Practice

This is a repo I use often to brush up my skills in Tensorflow and to learn and keep track of some minute things which often go unnoticed.

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

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## License
[MIT](https://choosealicense.com/licenses/mit/)
