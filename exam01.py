def f(x):
    return 2*x+1
x_value = [0,1,2,3]
for x in x_value:
    y=f(x)
    print(y)

from sympy import *
x = symbols('x')
f = 2*x+1
plot(f)
