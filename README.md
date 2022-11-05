# General Solutions

General solutions to nonhomogeneous linear equations Ax = b.

Any solution to Ax=b can be expressed as the sum of a special solution to Ax=b, plus a solution to Ax=0.

```python
import numpy as np
from general_solutions import gsolve, gss_usage
...
gss_usage()            # show the usage in detail
...
x,B,n = gsolve(A,b)    # get a special solution, a matrix of bases, the nullity of A
...
x1 = x + B.dot(v)      # get a new solution with a vector with length n 
...
```
