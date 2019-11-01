import torch
import torchvision
import matplotlib.pyplot as plt

cifar = torchvision.datasets.CIFAR10('.temp/', download=True)
# print(cifar[0])

fig = plt.figure(figsize=(1,1))
sub = fig.add_subplot(111)
sub.imshow(cifar[0][0])

# converting images to tensors

from torchvision import transforms

pipeline = transforms.Compose([transforms.ToTensor()])

cifar_tr = torchvision.datasets.CIFAR10('.temp/', transform=pipeline)

print(cifar_tr[0])

