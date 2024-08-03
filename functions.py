import math

sech = lambda x: 1 / math.cosh(x)

tanh = lambda x: math.tanh(x)
d_tanh = lambda x: sech(x) ** 2

sigmoid = lambda x: 1 / (1 + math.exp(-x))
d_sigmoid = lambda x: sigmoid(x) * (1 - sigmoid(x))

relu = lambda x: max(0, x)
d_relu = lambda x: 1 if x > 0 else 0 # Undefined at x = 0 but we can set it to 0

leaky_relu = lambda x: max(0.01 * x, x)
d_leaky_relu = lambda x: 1 if x > 0 else 0.01
