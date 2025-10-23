import doctest
import torch

def round_mantissa(x, decimals=1):
    # Avoid log(0) by adding a small epsilon
    eps = 1e-30
    exponent = torch.floor(torch.log10(torch.abs(x) + eps))
    
    # Handle zeros separately to avoid NaNs
    mantissa = x / (10 ** exponent)
    rounded_mantissa = torch.round(mantissa * (10 ** decimals)) / (10 ** decimals)
    
    return rounded_mantissa * (10 ** exponent)

def doctests(module, tests):
    """
    A helper function to combine unittest discovery with doctest suites.

    Args:
        module: The module to add doctests from.
        tests: The existing TestSuite discovered by unittest.

    Returns:
        A combined TestSuite with doctests added.
    """
    tests.addTests(doctest.DocTestSuite(module))
    return tests
