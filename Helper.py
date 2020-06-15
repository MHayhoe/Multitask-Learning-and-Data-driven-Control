import autograd.numpy as np


# Parametric linear-sigmoidal function
def sig_function(x, w, b=0):
    if len(np.shape(x)) == 3 and len(np.shape(w)) == 2:
        return sigmoid(np.einsum('ijk,ik->ij', x, w) + b[:,np.newaxis])
    elif len(np.shape(x)) == 3 and len(np.shape(w)) == 1:
        return sigmoid(np.einsum('ijk,k->ij', x, w) + b)
    else:
        return sigmoid(np.einsum('jk,k->j', x, w) + b)
    # shapes = np.shape(x)
    # num_counties = shapes[0]
    # num_vals = shapes[-1]
    # if len(shapes) == 3:
    #     return sigmoid(np.sum(x * np.reshape(w, (num_counties, 1, num_vals)), axis=2) + np.reshape(b, (num_counties,1)))
    # else:
    #     return sigmoid(np.sum(x * np.array(w), axis=1) + np.array(b))


# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Simple linear-sigmoidal function
simple_function = sigmoid


# Returns the value x such that sigmoid(x) = y, i.e., sigmoid^-1(y).
def inverse_sigmoid(y):
    return -np.log(1/y - 1)


# Soft-max function
def soft_max(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)