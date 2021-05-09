from typing import List, Tuple
import sympy as sp

def insertSymbolsAndComputeValue (expr: sp.Expr, symbols: List[sp.Symbol], values: sp.Matrix) -> sp.Expr:
    """ Simpely insert the values instead of the symbols such that symbols[i] gets replaced by values[i]"""
    e = expr
    # Loop through each symbol and insert its value into the expression
    for symbol, value in zip(symbols, values):
        e = e.subs(symbol, value)
    return e
