# gradients are a numerical representation of a derivative and can be positive or negative.

import torch

X = torch.rand(1, requires_grad=True)
Y = X + 1.0

def mse(Y):
	diff = 3.0 - Y
	return (diff * diff).sum() / 2

# the gradients on our X - that tells us in which direction we are off from the right answer
loss = mse(Y)
loss.backward()
print(X.grad)

# using a learning loop

learning_rate = 1e-3
for i in range(0, 10000):
	Y = X + 1.0
	loss = mse(Y)

	loss.backward()

	# the learning part, so we turn off the gradients from getting updated temporarily
	with torch.no_grad():
		# the gradient tells us which direction we are off, so we go in opposite direction
		X -= learning_rate * X.grad
		# and we zero out the gradients to get fresh values on each learning loop iteration
		X.grad.zero_()

print(X)

# we see this is an approximate answer
