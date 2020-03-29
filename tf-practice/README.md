# Tensorflow-Practice

This is a repo I use often to brush up my skills in Tensorflow and to learn and keep track of some minute things which often go unnoticed.

## Usage

You can run all the scripts/notebooks individually by having all the required dependencies installed in your machine. Please note that you should have all the other scripts in your same local directory in order for the imports to work properly.

## Calculating number of parameters in Feed-forward Neural Networks:

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

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## License
[MIT](https://choosealicense.com/licenses/mit/)
