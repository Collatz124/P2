# Martin Sig Nørbjerg ( 03/05/2021 )
from typing import List, Tuple
import math
import sympy as sp
from lagrange import insertSymbolsAndComputeValue

def hessianMatrix (f: sp.Expr, g: List[sp.Expr], x: sp.Matrix, n: int, m: int = None) -> Tuple[sp.Matrix, List[sp.Symbol]]:
    """ Computes the hessian matrix points of the lagrangian """
    if (m == None): m = len(g) # The number of constrains
    
    # Setup the lagrange function
    L = f + sum([sp.Symbol(f"u{i}") * g[i] for i in range(m)])
    # NOTE: The order is switches compared to bordered hessian matricies
    symbols = [x[i] for i in range(n)] + list(sp.symbols(" ".join(f"u{i}" for i in range(m))))
    
    # Compute the bordered hessian matrix
    H = sp.hessian(L, tuple(symbols))

    return H, symbols
    
def isSaddlepoint (H: sp.Matrix, m: int, n: int) -> bool:
    """ Figure out if the point (x, u) is a saddle point of the lagrangian """
    for k in range((m + n) // 2):
        # Test each submatrix (extract, takes a list of columns and rows and produces a submatrix)    
        if (H.extract(list(range(2 * (k + 1))), list(range(2 * (k + 1)))).det() < 0): return True
    return False

def testPoint (f: sp.Expr, g: List[sp.Expr], x: sp.Matrix, n: int, x0: List[float], u0: List[float], m: int = None) -> str:
    """ Tests a specific point of the lagrange function """
    if (m == None): m = len(g)
    
    # Compute the general hessian matrix
    H, symbols = hessianMatrix(f, g, x, n, m = m)
    
    # Insert values into the matrix and compute 
    specificHessian = insertSymbolsAndComputeValue(H, symbols, sp.Matrix(u0 + x0))

    # Check what kind of extrema x0 is.
    return f"f attains a saddlepoint at ({[round(x, 3) for x in x0]}, {[round(u, 3) for u in u0]})." if isSaddlepoint(specificHessian, m, n) else f"({[round(x, 3) for x in x0]}, {[round(u, 3) for u in u0]}) isn't a saddle point."
    
if (__name__ == "__main__"):
    n = 2
    x = sp.Matrix([sp.symbols(" ".join(["x" + str(i + 1) for i in range(n)]))])
    f = -x[0] * x[1]
    g1 = x[0] + x[1] * x[1] - 2
    g2 = -x[0]
    g3 = -x[1]
    print(testPoint(f, [g1, g2, g3], x, n, [4/3, math.sqrt(2/3)], [math.sqrt(2/3), 0, 0]))
    # Output:
    # f attains a saddlepoint at ([1.333, 0.816], [0.816, 0, 0]).