# Truss

This [python](https://www.python.org/) library can be used to optimize truss structures
using the Augmented Lagrangian Method [[1]](#Bir14a).

## Dependencies

* Ipopt [[2]](#Wäc06a)[[3]](#Wäc20a)
* python (> 3.6)
* numpy
* scipy
* numdifftools
* ipopt [[4]](#Küm20a)

## Install

`Truss` can be obtained by cloning this repository to the local system and running

```
pip3 install Truss/
```

in the command line.

## Usage

A minimal example can be found in the [Example](Example) directory.

## References

<a name='Bir14a'>[1]</a> E.G. Birgin and J.M. Martínez, Practical Augmented Lagrangian Methods for Constrained Optimization (Society for Industrial and Applied Mathematics (SIAM), Philadelphia, 2014).
<a name='Wäc06a'>[1]</a> A. Wächter and L.T. Biegler, Math. Program. **106**, 25 (2006).
<a name='Wäc20a'>[1]</a> A. Wächter and L.T. Biegler (2020), ['Ipopt'](https://github.com/coin-or/Ipopt).
<a name='Küm20a'>[1]</a> M. Kümmerer (2020), ['cyipopt'](https://github.com/matthias-k/cyipopt).