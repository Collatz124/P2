#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 19:06:39 2021

@author: amaliepoulsen
"""

import numpy as np
from scipy.optimize import minimize

def objective(x):
    return -x[0]**2*x[1]**2

def constraint1(x):
    return -1.39*x[0]-0.55*x[1]+1500

def constraint2(x):
    return -14*x[0]-6.5*x[1]+20000

def constraint3(x):
    return x[0]+x[1]-2000

def constraint4(x):
    return x[0]

def constraint5(x):
    return x[1]

# initial guesses
n = 2
x0 = np.zeros(n)
x0[0] = 600.0
x0[1] = 1000.0

# show initial objective
print('Initial SSE Objective: ' + str(objective(x0)))

# optimize
con1 = {'type': 'ineq', 'fun': constraint1} 
con2 = {'type': 'ineq', 'fun': constraint2}
con3 = {'type': 'ineq', 'fun': constraint3}
con4 = {'type': 'ineq', 'fun': constraint4}
con5 = {'type': 'ineq', 'fun': constraint5}
cons = ([con1,con2,con3,con4,con5])
solution = minimize(objective,x0,method='trust-constr',\
                    constraints=cons)
x = solution.x

# show final objective
print('Final SSE Objective: ' + str(objective(x)))

# print solution
print('Solution')
print('x= ' + str(x[0]))
print('y= ' + str(x[1]))
