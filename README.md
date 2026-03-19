# BoxDMK.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ArrogantGao.github.io/BoxDMK.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ArrogantGao.github.io/BoxDMK.jl/dev/)
[![Build Status](https://github.com/ArrogantGao/BoxDMK.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ArrogantGao/BoxDMK.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ArrogantGao/BoxDMK.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ArrogantGao/BoxDMK.jl)

A Julia implementation of the **Box-based Data-driven kernel-independent Method of Kernels (BoxDMK)** for fast evaluation of volume potentials on adaptive hierarchical box trees in 3D.

Given a source density $f$ defined on a domain $\Omega$ and a kernel $K$ (e.g., Laplace, Yukawa), BoxDMK rapidly computes the volume potential:

$$u(\mathbf{x}) = \int_\Omega K(\mathbf{x}, \mathbf{y})\, f(\mathbf{y})\, d\mathbf{y}$$

along with optional gradients and Hessians, to user-specified accuracy.

## Features

- **Kernel-independent**: supports Laplace ($1/r$), Yukawa ($e^{-\beta r}/r$), and square-root Laplacian ($1/r^2$) kernels via Sum-of-Gaussians decomposition
- **Adaptive refinement**: error-driven tree construction with level-restricted balancing
- **Near-linear complexity**: $O(N)$ evaluation via Box Fast Gauss Transform + hierarchical plane wave expansions
- **Derivatives**: optional gradient and Hessian computation at no extra algorithmic cost
- **Target evaluation**: interpolate results to arbitrary points in the domain
- **Multiple bases**: Legendre and Chebyshev polynomial discretizations
- **Threaded**: parallelism via `Threads.@threads` in all major passes

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/ArrogantGao/BoxDMK.jl")
```

Requires Julia 1.10 or later.

## Quick Start

```julia
using BoxDMK

# Define a source density function: f(x) → Vector of length nd
f(x) = [exp(-100 * sum((x .- 0.5).^2))]

# Build adaptive tree
tree, fvals = build_tree(f, LaplaceKernel(), LegendreBasis();
    ndim=3, norder=8, eps=1e-6, boxlen=1.0, nd=1)

# Compute volume potential
result = bdmk(tree, fvals, LaplaceKernel(); eps=1e-6)

# Access the potential on the tensor grid
result.pot  # Array{Float64, 3} of shape (nd, norder^3, nboxes)
```

### With gradients and target evaluation

```julia
# Compute potential + gradient + Hessian
result = bdmk(tree, fvals, LaplaceKernel();
    eps=1e-6, grad=true, hess=true)

result.grad  # (nd, 3, norder^3, nboxes)
result.hess  # (nd, 6, norder^3, nboxes) — xx, yy, zz, xy, xz, yz

# Evaluate at arbitrary target points
targets = rand(3, 100)  # 100 random points in 3D
result = bdmk(tree, fvals, LaplaceKernel();
    eps=1e-6, targets=targets)

result.target_pot  # (nd, 100)
```

### Supported kernels

```julia
LaplaceKernel()        # 1/(4πr)
YukawaKernel(β)        # exp(-βr)/(4πr)
SqrtLaplaceKernel()    # 1/(4πr²)
```

### Supported bases

```julia
LegendreBasis()        # Gauss-Legendre nodes
ChebyshevBasis()       # Chebyshev nodes
```

## Algorithm

BoxDMK uses a 9-step pipeline:

1. **Precomputation** — Sum-of-Gaussians decomposition, proxy system setup, interaction list construction
2. **Taylor corrections** — local corrections for the SOG approximation residual
3. **Upward pass** — density to proxy charges, anterpolation up the tree
4. **Multipole plane wave formation** — proxy charges to complex PW expansions
5. **M2L translation** — plane wave shift between well-separated boxes
6. **Downward pass** — local PW to proxy potential, interpolation down the tree
7. **Direct local interactions** — near-field evaluation via precomputed sparse tables
8. **Asymptotic expansion** — analytical treatment of small-variance Gaussian components
9. **Output assembly** — proxy potential to user grid, derivative computation

The algorithm achieves $O(N)$ complexity by decomposing any radial kernel as a sum of Gaussians, then using the Box Fast Gauss Transform at each level of an adaptive tree hierarchy.

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ndim` | Spatial dimension (currently 3 only) | `3` |
| `norder` | Polynomial order per dimension | `6` |
| `eps` | Target accuracy | `1e-6` |
| `boxlen` | Root box side length | `1.0` |
| `nd` | Number of right-hand sides | `1` |
| `grad` | Compute gradient | `false` |
| `hess` | Compute Hessian | `false` |
| `targets` | Evaluation points (ndim × ntarg matrix) | `nothing` |

## Acknowledgments

This package is a Julia translation of the [Fortran BoxDMK library](https://github.com/flatironinstitute/boxdmk), developed at the Flatiron Institute. The algorithm is based on:

- The Box Fast Gauss Transform for kernel-independent fast summation
- Data-driven Sum-of-Gaussians kernel approximation
- Adaptive hierarchical box tree discretization

## License

MIT License. See [LICENSE](LICENSE) for details.
