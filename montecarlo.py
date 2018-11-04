from math import *
from numpy import *
from random import *


# Density_Function of the standard gaussian distribution
def density_normal(x):
  return 1/sqrt(2*pi) * exp(-x**2 / 2)

#This would be the uniform density function. But do not need this actually, I think
def density_uniform(x):
  if (-5 <= x and x <= 5):
    return 1/10
  else:
    return 0

# Density involving the cosinus:
def density_cosinus(x):
  if (-1 <= x and x <= 1):
    return (1 + cos(pi*x)) / 2
  else:
    return 0
