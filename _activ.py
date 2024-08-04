import math

# sech = lambda x: 1 / math.cosh(x)

tanh = lambda x: math.tanh(x)
d_tanh = lambda output: 1 - output ** 2

sigmoid = lambda x: 1 / (1 + math.exp(-x))
d_sigmoid = lambda output: output * (1 - output)

relu = lambda x: max(0, x)
d_relu = lambda output: 1 if output > 0 else 0

leaky_relu = lambda x: max(0.01 * x, x)
d_leaky_relu = lambda output: 1 if output > 0 else 0.01