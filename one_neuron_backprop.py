import numpy as np
# input values
w = [-3.0, -1.0, 2.0]  # weights
x = [1.0, -2.0, 3.0]
b = 1.0  # bias

# Multiplying inputs by weights
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]

# Adding weighted inputs and a bias
z = xw0 + xw1 + xw2 + b
print(z)
# ReLU activation function
y = max(z, 0)

# Backward pass
# The derivative from the n`ext layer
dcost_drelu = 1.0

# Derivative of ReLU and the chain rule
drelu_dz = 1.0 if z > 0 else 0.0
dcost_dz = dcost_drelu * drelu_dz

dz_dw0 = x[0]
dz_dw1 = x[1]
dz_dw2 = x[2]
dz_db = 1

dcost_dw0 = dcost_dz * dz_dw0
dcost_dw1 = dcost_dz * dz_dw1
dcost_dw2 = dcost_dz * dz_dw2

dz_dx0 = w[0]
dz_dx1 = w[1]
dz_dx2 = w[2]

dcost_dx0 = dcost_dz * dz_dx0
dcost_dx1 = dcost_dz * dz_dx1
dcost_dx2 = dcost_dz * dz_dx2

dcost_db = dcost_dz * dz_db

dx = np.array([dcost_dx0, dcost_dx1, dcost_dx2])
dw = np.array([dcost_dw0, dcost_dw1, dcost_dw2])
db = dcost_db

# then optimizer role
learning_rate = 0.001
w = w-dw*learning_rate
b = b-db*learning_rate
x = x-dx*learning_rate

newz = np.dot(w, x)+b
print(newz)
print(w, '\n', b, '\n', x)
