import sympy as sp
from typing import List, Tuple
from misc import insertSymbolsAndComputeValue

def newtonRapson (gradient: sp.Matrix, x: sp.Matrix, symbols: List[sp.Symbol], n: int, m: int, iterations: int = 10, negativeAllowed: bool = True):
    """ Perform newton-rapson iteration """ # This will be used to compute the critical points
    # Compute the general jacobian matrix
    jacobian = sp.Matrix([[sp.diff(gradient[i], symbol) for symbol in symbols] for i in range(n + m)])
    
    for _ in range(iterations):
        # Check if the value was negative
        if (negativeAllowed == False):
            for value in x[:n]:
                if (value < 0):
                    raise ValueError("A component of x was negative")
                
        # Compute the gradient & the jacobian
        g = insertSymbolsAndComputeValue(gradient, symbols, x)
        j = insertSymbolsAndComputeValue(jacobian, symbols, x)

        if (j.det() == 0):
            raise ValueError("Determinant in jacobian matrix was 0!")
        
        # Perform the actual iteration
        x -= j.inv() * g
        
    return x