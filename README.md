# Truss

This [python](https://www.python.org/) library can be used to optimize truss structures
using the Augmented Lagrangian Method [[1]](#Bir14a).

## Dependencies

* Ipopt [[2]](#Wäc06a)[[3]](#Wäc20a)
* python (> 3.6)
* numpy
* scipy
* numdifftools [[4]](#Bro20a)
* ipopt [[5]](#Küm20a)

## Install

`Truss` can be obtained by cloning this repository to the local system and running

```
pip3 install Truss/
```

in the command line.

## Usage

A minimal example can be found in the [Example](Example) directory.

## References

<a name='Bir14a'>[1]</a> E.G. Birgin and J.M. Martínez, Practical Augmented Lagrangian Methods for Constrained Optimization (Society for Industrial and Applied Mathematics (SIAM), Philadelphia, 2014).<br/>
<a name='Wäc06a'>[2]</a> A. Wächter and L.T. Biegler, Math. Program. **106**, 25 (2006).<br/>
<a name='Wäc20a'>[3]</a> A. Wächter and L.T. Biegler (2020), ['Ipopt'](https://github.com/coin-or/Ipopt).<br/>
<a name='Bro20a'>[4]</a> P.A. Brodtkorb (2020), ['numdifftools'](https://github.com/pbrod/numdifftools).<br/>
<a name='Küm20a'>[5]</a> M. Kümmerer (2020), ['cyipopt'](https://github.com/matthias-k/cyipopt).<br/>