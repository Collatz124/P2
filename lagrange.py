# Martin Sig Nørbjerg ( 23/04/2021 )
from typing import List, Tuple
import sympy as sp
from misc import insertSymbolsAndComputeValue
from sympy.solvers.solveset import nonlinsolve
from solver import newtonRapson

def lagrange (f: sp.Expr, g: List[sp.Expr], x: sp.Matrix, n: int, m: int = None, numberOfIterationPoints: int = 10) -> List[Tuple[float]]:
    """ Computes the critical points of the lagranian """
    if (m == None): m = len(g) # The number of constrains
    
    # Setup the lagrange function
    L = f + sum([sp.Symbol(f"u{i}") * g[i] for i in range(m)])
    symbols = [x[i] for i in range(n)] + [sp.symbols(" ".join(f"u{i}" for i in range(m)))]
    
    equations = [sp.diff(L, s) for s in symbols]
    print(newtonRapson(sp.Matrix(equations), sp.Matrix([0.1 for _ in range(m + n)]), symbols, m = m + n))
    return nonlinsolve(equations, symbols)

if (__name__ == "__main__"):
    n = 2
    x = sp.Matrix([sp.symbols(" ".join(["x" + str(i + 1) for i in range(n)]))])
    f = x[0] * x[0] + x[1] * x[1] - 2*x[0] - 2*x[1]
    g1 = x[0] * x[0] + x[1] * x[1] - 4
    print(lagrange(f, [g1], x, n))
