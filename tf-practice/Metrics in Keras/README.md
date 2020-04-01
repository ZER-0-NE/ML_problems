## Metrics in Keras

This repo defines and derives some of the metrics that are used in Keras along with some info about its mathematical background.


### Why do we need Activation Function?

<p align="center">
  <img  src=/assets/why_activ_func.png/>
</p>

In a Neural Network every neuron will do two computations:

a)Linear summation of inputs: If we see the above diagram , it has two inputs x1,x2 and a bias(b). We have weights w1 and w2.

`sum=(w1*x1+w2*x2)+b`

b) Activation computation: This computation decides, whether a neuron should be activated or not, by calculating weighted sum and further adding bias with it. **The purpose of the activation function is to introduce non-linearity into the output of a neuron.**

#### Why do we need Non-linear activation functions :-

A neural network without an activation function is essentially just a linear regression model. The activation function does the non-linear transformation to the input making it capable to learn and perform more complex tasks.

#### Activation Functions Cheat-sheet

<p align="center">
  <img  src=/assets/activ_func_cheat.png/>
</p>