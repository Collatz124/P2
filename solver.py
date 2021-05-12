import sympy as sp
from typing import List, Tuple
from misc import insertSymbolsAndComputeValue

def newtonRapson (gradient: sp.Matrix, x: sp.Matrix, symbols: List[sp.Symbol], m: int, iterations: int = 10):
    """ Perform newton-rapson iteration """ # This will be used to compute the critical points
    # Compute the general jacobian matrix
    jacobian = sp.Matrix([[sp.diff(gradient[i], symbol) for symbol in symbols] for i in range(m)])
    # sp.pprint(jacobian)
    for _ in range(iterations):
        # Compute the gradient & the jacobian
        print(_)
        g = insertSymbolsAndComputeValue(gradient, symbols, x)
        j = insertSymbolsAndComputeValue(jacobian, symbols, x)

        if (j.det() == 0):
            raise ValueError("Determinant in jacobian matrix was 0!")
        
        # Perform the actual iteration
        x -= j.inv() * g
        x = sp.simplify(x)
    return x