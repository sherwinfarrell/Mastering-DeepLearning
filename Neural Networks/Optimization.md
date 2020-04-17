# A Few Optimization Algorithms

## Gradient Descent Algorithms

# Mini Batch Gradient Descent

Here the batch of training examples are broken down into multiple mini batches which are then trained.
This allows for faster learning as each epoch ( A single computation done by NN on all the training examples) sees multiple gradient descents unlike that scene in Batch Gradient Descent (All the training examples as one batch). There is also ** Stochastic Gradient Descent ** which is a mini batch made up of only one training example. According to the course the mini batch fairs well in optimizing gradient descent as it takes advantage of vectorization and at the same time doesn't fall the expense of expensive computation resulting from batch gradient descent trying to compute the large number of training examples at once.

The code for Mini Batch:
```python
permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k*mini_batch_size : (k+1) *mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k*mini_batch_size : (k+1) *mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, (k+1)*mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, (k+1)*mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
```

The forward and backward computation for Stochastic Gradient Descent:
```python
X = data_input
Y = labels
parameters = initialize_parameters(layers_dims)
for i in range(0, num_iterations):
    for j in range(0, m):
        # Forward propagation
        a, caches = forward_propagation(X[:,j], parameters)
        # Compute cost
        cost += compute_cost(a, Y[:,j])
        # Backward propagation
        grads = backward_propagation(a, caches, parameters)
        # Update parameters.
        parameters = update_parameters(parameters, grads)
```