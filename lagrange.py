# Martin Sig Nørbjerg ( 23/04/2021 )
from typing import List, Tuple
import sympy as sp
from solver import newtonRapson
from misc import insertSymbolsAndComputeValue
from bordered import testPoint
import random

def isCloseToZeroVector (vector: sp.Matrix, threshHold: float = 0.01) -> bool:
    """ Checks that the vector is close to a vector of zeros """
    for val in vector:
        if (val >= threshHold): return False
    return True

def isZeroVector (vector: sp.Matrix) -> bool:
    """ Checks if the vector is a zero vector """
    for val in vector:
        if (val != 0): return False
    return True

def lagrange (f: sp.Expr, g: List[sp.Expr], x: sp.Matrix, n: int, m: int = None, numberOfIterationPoints: int = 20, numberOfIterations: int = 10, allowNegative: bool = False, threshHold: float = 0.001) -> List[Tuple[float]]:
    """ Computes the critical points of the lagranian """
    if (m == None): m = len(g) # The number of constrains
    
    # Setup the lagrange function
    L = f + sum([sp.Symbol(f"u{i}") * g[i] for i in range(m)])
    symbols = [x[i] for i in range(n)] + [sp.Symbol(f"u{i}") for i in range(m)]

    gradient = sp.Matrix([sp.diff(L, s) for s in symbols])
    
    # Find several different candidates 
    criticalPointCandidates = []
    for _ in range(numberOfIterationPoints):
        # The range can be changed to better match the problem.
        point = sp.Matrix([random.uniform(0, 10) for _ in range(m + n)])
        
        try:
            criticalPointCandidates.append(newtonRapson(gradient, point, symbols, n = n, m = m, iterations = numberOfIterations, negativeAllowed = allowNegative))
        except ValueError:
            pass
    
    print(f"found {len(criticalPointCandidates)} candidate points.")
    
    # Remove similar points & non critical points
    points = []
    for point in criticalPointCandidates:
        # Compute gradient at point
        gradientAtPoint = insertSymbolsAndComputeValue(gradient, symbols, point)
        
        # If gradient isn't a critical point perform more iterations 
        if (isCloseToZeroVector(gradientAtPoint, threshHold = threshHold) == False):
            try:
                point = newtonRapson(gradient, point, symbols, n = n, m = m, iterations = numberOfIterations, negativeAllowed = allowNegative)
                gradientAtPoint = insertSymbolsAndComputeValue(gradient, symbols, point)
                break
            
            except ValueError:
                pass

        # Check that the gradient is atleast similar to the 0 vector
        if (isCloseToZeroVector(gradientAtPoint, threshHold = threshHold) == True):
            # Check if there is already a similar point otherwise append the point the points list 
            if (len(points) == 0):
                points.append(point)
            else:
                for idx, p in enumerate(points):
                    difference = p - point
                    if (sp.sqrt(difference.dot(difference)) <= threshHold):
                        break
                    
                    if (idx == len(points) - 1):
                        points.append(point)
                        break # This is needed otherwise the length will just increase by one and the loop will continue

                
    # Classify the points (Here we assume that every point is in fact a critical point
    #                      Even though we don't nessecarily know)
    classificationOfPoints = []
    for point in points:
        gradientAtPoint = insertSymbolsAndComputeValue(gradient, symbols, point)       
        classificationOfPoints.append(testPoint(f, g, x, n, point[:n], point[n:]))

    return classificationOfPoints

if (__name__ == "__main__"):
    n = 2
    x = sp.Matrix([sp.symbols(" ".join(["x" + str(i + 1) for i in range(n)]))])
    f = 160 * x[0] + 250 * x[1] # Minimize this function
    g1 = 20 * sp.sqrt(x[0]) * sp.sqrt(x[1]) - 1000 # Under this constraint
    print(lagrange(f, [g1], x, n, allowNegative = False))
    # Output
    # found 13 candidate points.
    # ['f atains a local minimum at [62.5000000000000, 40.0000000000000], when restriced to candidate points.']