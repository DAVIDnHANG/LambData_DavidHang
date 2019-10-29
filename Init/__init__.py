"""
a collection of data science helper function
"""

import pandas as pd
import numpy as np

#sample dataset
ONES = pd.DataFrame(np.ones(10))
ZEROS = pd.DataFrame(np.zeros(50))

#sample functions
def increment(x):
    return(x+1)
