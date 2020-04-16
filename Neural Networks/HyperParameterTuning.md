# This Page will discuss a few Hyperparameters that can be tuned for NN optimization


## Layer Initialization
### Zero Initialization
```python
    for l in range(1, L):
        parameters['W' + str(l)] = np.zeros([layers_dims[l],layers_dims[l-1]])
        parameters['b' + str(l)] = np.zeros([layers_dims[l],1])
    return parameters
```

<font color='yellow'>

**What you should remember**:
- The weights W[l] should be initialized randomly to break symmetry. 
- It is however okay to initialize the biases b[l] to zeros. Symmetry is still broken so long as W[l] is initialized randomly.

</font>

### Initializing Randomly

```python
for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros([layers_dims[l],1])
```

### HE Initialization:

```python
for l in range(1, L + 1):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1] * 
                                   (2/np.sqrt(layer_dims[l-1]))
        parameters['b' + st) * r(l)] = np.zeros([layer_dims[l],1])
```

### Xavier Initialization:

```python
for l in range(1, L + 1):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1] * 
                                   (np.sqrt(1/layer_dims[l-1]))
        parameters['b' + st) * r(l)] = np.zeros([layer_dims[l],1])
```


## Regularization

Overfitting is generally results from a lack in training data. However there are steps to reduce overfitting using a few techniques. It reduces the effect of the weights by either reducing them in magnitude or by by completely switching off certain weights. It in a sense brings to a more basic Neural network, depending of the effect of the regularization parameters.

### L2 Regularization

Assuming there are 3 layers the L2 regularization cost is as follows:

```python
L2_regularization_cost = ((1/m)*(lambd/2)*(np.sum(np.square(W1)) + np.sum(np.square(W2)) + 
                          np.sum(np.square(W3))))
```
<font color="maroon">
This has to be added to the actual cost and together they bring about a regularization effect when training the weights using gradient descent.
</font>

The gradient thus for the weights become:
```python
    dW3 = 1./m * np.dot(dZ3, A2.T) + (lambd/m)*W3
```
Since dW3 is increased by a factor of Lamda/m * W3 there is a (1 - lambda/m) reduction in W3 when the weight is updated using W3 - alpha * dW3