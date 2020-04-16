# Getting Started with the Basics

## Working With Matrices
The images are 64x64x64 pixel matrices that are correspond to single matrices of red, blue and green.
To unroll it into a feature matrix we need to collect all values of red and similarly concatenate blue and green after that until dimension is 12288. 
nx or n = 12288

We use 1 or 0 for true or false in the label.

If we use m training examples then the shape of the traning set will be Nx x m ( m -> number of training values and nx is the number of pixels per training example)
And the results set denoted as Y is will be stacked column wise and the Y.shape = (1, m)

##Logistic Regression
>Here y = P(y = 1| x)
x = nx dimension vector
Parameters:
Weight: Nx Dimension
Bias = 1 Dimension

We use sigmoid to get the wT x + b to be within 1 and 0 
>wT x + b = Z
>sigmoid(z) = 1/(1-e^-z)

> if z large = sigmoid = 1/(1+0) == 1
> if z is small sigmoid = 1/(1 + Bigg number  )  == 0


## We now check what cost functions are and how to go about implementing them
Also know as error function

Optimization problem if we use 1/2(ypred - y)^2
We need to find a global maximum 

We use the following log error function: 
> -(ylog(ypred) + (1-y)log(1-yped))

if y = 1: L(ypred, y) = -log(ypred)        Want log(ypred) large and hence we want ypred large
if y = 0: L(ypred, y) = -log(1-ypred)      Want log(1-pred) large and hence we want ypred small

This is for a single training sample and the cost measures how your entire taining set is doing.
> COst function : J(w,b) = 1/m(SUM(L(ypred(i),y(i))))

> -1/m(SUM((ylog(ypred) + (1-y)log(1-yped)))

```python
logprobs = np.multiply(-np.log(A3),Y) + np.multiply(-np.log(1 - A3), 1 - Y)
cost = 1./m * np.sum(logprobs)
```

##Gradient Descent:
Gradient descent is usually done with the help of the cost function and not on the loss or the error function.

We use w := w - alpha d(J(w))/dw  This is only for a cost function depending on W

But since we use a cost function dependent of both w and b as Ypred depends on w and b we use the following:
> w:= w - alpha d(J(w,b))/dw  Where the derivate is a partial derivate of J(w,b) with respect to w
> b:= w - alpha d(J(w,b))/db 

