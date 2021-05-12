# Martin Sig Nørbjerg ( 10/05/2021 )
from typing import List, Tuple
import sympy as sp
from lagrange import insertSymbolsAndComputeValue

def borderedHessianMatrix (f: sp.Expr, g: List[sp.Expr], x: sp.Matrix, n: int, m: int = None) -> Tuple[sp.Matrix, List[sp.Symbol]]:
    """ Computes the bordered-hessian matrix points of the lagranian """
    if (m == None): m = len(g) # The number of constrains
    
    # Setup the lagrange function
    L = f + sum([sp.Symbol(f"u{i}") * g[i] for i in range(m)])
    # NOTE: This is the other way around in this progam (because the order is important in the hessian matrix)
    symbols = [sp.Symbol(f"u{i}") for i in range(m)] + [x[i] for i in range(n)]
    
    # Compute the bordered hessian matrix
    H = sp.hessian(L, tuple(symbols))
    return H, symbols
    
def isMaximum (H: sp.Matrix, m: int, n: int) -> bool:
    """ Figure out if the point (x, u) is a local maximum """
    for k in range(2*m + 1, m + n + 1):
        # Test each submatrix (extract, takes a list of columns and rows and produces a submatrix)
        if ((-1)**(k - m) * H.extract(list(range(k)), list(range(k))).det() <= 0): return False
    return True

def isMinimum (H: sp.Matrix, m: int, n: int) -> bool:
    """ Figure out if the point (x, u) is a local minimum """
    for k in range(2*m + 1, m + n + 1):
        # Test each submatrix (extract, takes a list of columns and rows and produces a submatrix)
        if ((-1)**(m) * H.extract(list(range(k)), list(range(k))).det() <= 0): return False
    return True

def isSaddlepoint (H: sp.Matrix, m: int, n: int) -> bool:
    """ Figure out if the point (x, u) is a saddle point """
    for k in range(2*m + 1, m + n + 1):
        # Test each submatrix (extract, takes a list of columns and rows and produces a submatrix)
        if (H.extract(list(range(k)), list(range(k))).det() == 0): return False
    return True

def testPoint (f: sp.Expr, g: List[sp.Expr], x: sp.Matrix, n: int, x0: List[float], u0: List[float], m: int = None) -> str:
    """ Tests a specific point of the lagrange function """
    if (m == None): m = len(g)
    
    # Compute the general hessian matrix
    H, symbols = borderedHessianMatrix(f, g, x, n, m = m)
    
    # Insert values into the matrix and compute 
    specificHessian = insertSymbolsAndComputeValue(H, symbols, sp.Matrix(u0 + x0))

    # Check what kind of extrema x0 is.
    if (isMinimum(specificHessian, m, n) == True): return f"f atains a local minimum at {x0}, when restriced to candidate points."
    elif (isMaximum(specificHessian, m, n) == True): return f"f atains a local maximum at {x0}, when restriced to candidate points."
    elif (isSaddlepoint(specificHessian, m, n) == True): return f"f has a local saddle point at {x0}, when restricted to candidate points." 
    else: return "Test was inconclusive"    
    
if __name__ == "__main__":
    n = 3
    x = sp.Matrix([sp.symbols(" ".join(["x" + str(i + 1) for i in range(n)]))])
    f = x[0] * x[0] + x[1] * x[1] + x[2] * x[2]
    g1 = x[0] * x[0] + x[1] * x[1] - x[2] * x[2]
    g2 = x[0] - 2 * x[2] - 6
    print(testPoint(f, [g1, g2], x, n, [-6, 0, -6], [-3, -24]))
    print(testPoint(f, [g1, g2], x, n, [2, 0, -2], [-1 / 3, -8 / 3]))
    # Output:
    # f atains a local maximum at [-6, 0, -6], when restriced to candidate points.
    # f atains a local minimum at [2, 0, -2], when restriced to candidate points.
