# Exploration of AdderViT

## Motivation

Recently, most of the neural networks computation are based on matrix multiplication. However, matrix multiplication itself is a computationally expensive operation. 

Since the dot product of vectors (the basics of matrix multiplication) essentially calculates the dot product similarity (the unnormalized cosine similarity), we wondered if we could **find a more concise alternative to the dot product similarity to speed up inference**.

There's already some works doing something similar. For example, [AdderNet](https://arxiv.org/abs/1912.13200) used $l1$ distance to calculate the similarity between the convolution kernels and feature maps; [ShiftAddNet](https://proceedings.neurips.cc/paper/2020/file/1cf44d7975e6c86cffa70cae95b5fbb2-Paper.pdf) tried quantize the multiplication to $\times2^n$ using a shift operation on fixed-point number; [EuclidNet](https://arxiv.org/abs/2212.11803) implemented $l2$ distance to improve the performance. 

But all the works above were applied on CNN, so we try to transfer this sort of method to transformer-based model. 

## Adder-FC

1. Multi: $Y_{L,D_{out}}=X_{L,D_{in}}\cdot W_{D_{in},D_{out}}+b_{D_{out}}$ in which $y_{l,d}=\sum\limits_{i=1}^{D_{in}} x_{l,i} \times w_{i,d}$  
2. Adder: $Y_{L,D_{out}}=X_{L,D_{in}}\bigoplus W_{D_{in},D_{out}}+b_{D_{out}}$ in which $y_{l,d}=-\sum\limits_{i=1}^{D_{in}}| x_{l,i} - w_{i,d}|$  
3. Euclid: $Y_{L,D_{out}}=X_{L,D_{in}}\bigotimes W_{D_{in},D_{out}}+b_{D_{out}}$ in which $y_{l,d}=-\sum\limits_{i=1}^{D_{in}}( x_{l,i} - w_{i,d})^2$

### Problem 1: Back propagation on Adder

We compared these methods in a Simple_ViT model. Method 1 and 3 are derivable function, so back propagation can be applied directly. But the Adder method has the problem with back propagation. Following addernet, we used the full precision gradient to replace the sign precision gradient. (Get more information from [AdderNet](https://arxiv.org/abs/1912.13200))

$\frac{\partial y}{\partial w} = sign(x-w) \space\space\longrightarrow\space\space \frac{\partial y}{\partial w} = x-w$

$\frac{\partial y}{\partial x} = sign(w-x) \space\space\longrightarrow\space\space \frac{\partial y}{\partial x} = HardTanh(w-x)$

But we will get a training error showed on chart bellow: Both the train and val accuracy slide down. Note that according to the back propagation method above, for W, addition and multiplication eventually converge to the same place. However, for X, although the gradient clipping is applied, there is still a situation where the forward and back propagation are not unified, which will affect the gradient that the rebate propagation continues to propagate, thus leading to incorrect updates.

![training_error](./pictures/train_error.png)

So we change the way we backpropagate of X to be closer to the gradient of the absolute value.

$\frac{\partial y}{\partial x} = sign(w-x) \space\space\longrightarrow\space\space \frac{\partial y}{\partial x} = HardTanh(4\times(w-x))$

This simple $\times4$ can cause more derivative clamped, leading the full precision gradient more close to the true gradient. (It shows as pictures below. The graph shows the original function corresponding to each gradient function.)

![gradient_illu](./pictures/gradient_illu.png)

In this way, AdderViT can be trained normally. (We also tried sign derivative, but it leads to  slightly accurate lose.)

### Problem 2: $l1$-distance caused dimensionality reduction

![distance_ex](./pictures/distance_ex.jpg)

We thought about how radial distances (distances in each norm) behave in space. Points on the same surface will map to the same number. When $p=1$ some area in the space may be mapped to the same point, which will lead to unreasonable dimensionality reduction and loss of information.

$X_{L,D}=(E_{D}\times X_{L,D})\bigoplus W_{D,D}+b_{D}$

We try to apply a transformation to the input so as to stretch the shape of the distribution of points in space to avoid unreasonable dimensionality reduction caused reasons above. (Note that this is a point-wise multiplication, so it doesn't cause too much computation.)

### Problem 3: Invalid residual connections

Residual connection is a very common operation. The basic idea is that if this layer can't learn anything useful, then let this layer be discarded and the information from the previous layer can be directly passed on without any loss of information. But in Adder operation, there will always be some information, even useless or poisonous, added into feature map. 

$X_{L,D}=X_{L,D}\bigoplus W_{D,D}+b_{D}\space\space$ (Cannot be satisfied)

The reason of essence is that we cannot find a suitable $W$ to complete the following identity mapping. (See [AdderSR](https://arxiv.org/abs/2009.08891) for the proof)

$X_{L,D}=F_{D}\times (X_{L,D}\bigoplus W_{D,D}+b_{D})$ 

So we introduce a soft-mask transformation $F$ to mask some of the output number. If you want an identity mapping, just make all element in $F$ to $0\space$.

### Final Methods

We use this function for forward propagation.

$X_{L,D}=F_{D}\times [(E_{D}\times X_{L,D})\bigoplus W_{D,D}+b_{D}]$

And this function for back propagation.

$\frac{\partial y}{\partial w} = x-w$

$\frac{\partial y}{\partial w} = HardTanh(4\times(w-x))$

But unfortunately, it seems that this additive network still has some unknown problems and does not get good results.

## Experiments

### Simple_ViT / ResNet20(AdderNet) on CIFAR10  

In this group of experiment, we used weak data augmentation, which makes the Multi overfits, which should actually be better than Adder.

``qkv:`` Change to_qkv linear transformation to Adder operation.  
``1ffn:`` Change 2 layers of feed forward to 1 layer of Adder operation.  
``2ffn:`` Change 2 layers of feed forward to 2 layer of Adder operation.  
``adder:`` Change FFN and to_qkv with original Adder operation from AdderNet.    
``multi:`` The control group for the original multiplication.   
``noupdtm:`` No Adaptive learning rate from AdderNet.   
``euclider:`` The $l2$-norm distance method.  
``wopout:`` Without project_out layer.  

![first_exp](./pictures/first_exp.png)
![final_result](./pictures/final_result.png)

### Simple_ViT on ImageNet

Due to the lack of underlying optimization in the training process of addition algorithm, the training speed is very slow. So it has only been trained on Imagenet for 25 epochs, but it has already opened a significant gap, and the fit of addition is significantly worse than that of multiplication.

![secend_exp](./pictures/secend_exp.png)

### Swin(weakened) on CIFAR10

No matter what model, Adder is always about 10% less accurate than Multi. In addition to the training difficulties caused by no optimization, our work was forced to pause.

![third_exp](./pictures/third_exp.png)

# Conclusion

1. The three problems mentioned above, and their solutions.
2. Adder is more difficult to train than Multi and can easily lead to unstable training situations.
3. There are still some unknown obstacles to the application of addition in full connection or attention, especially for large data sets.

# Note

Using the pycuda package, we optimized the new addition, which is faster than performing it directly in python, but not so much.

We made use of the [adder_cuda](https://github.com/LingYeAI/AdderNetCUDA) package, which is a cuda optimization of AdderNet. But you may meet some problem when install this package. Just be patient.

We still think Adder is a promising alternative to matrix multiplication. If you have any research on this topic or have any questions, please feel free to contact us.