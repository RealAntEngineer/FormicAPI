/**
 * Mathematical primitives used by FormicAPI.
 *
 * <p>This package provides the numerical foundation used by simulation
 * frameworks. It contains generic mathematical structures and algorithms for
 * representing, assembling, and solving numerical problems.
 *
 * <p>The package is intentionally independent from simulation concepts:
 * it does not reference nodes, components, materials, or physical domains.
 * Higher-level simulation code builds on these primitives to implement
 * physical models.
 *
 * <h2>Sub-packages</h2>
 *
 * <dl>
 *   <dt>{@link com.rae.formicapi.foundation.math.operators}</dt>
 *   <dd>
 *     Numerical operators and data structures used to represent mathematical
 *     transformations.
 *
 *     <p>This package is divided into:
 *     <ul>
 *       <li>
 *         {@code operators.linear} —
 *         Linear operators and matrix representations:
 *         <ul>
 *           <li>{@code Matrix} / {@code MutableMatrix} — contracts for
 *               linear transformations and matrix assembly.</li>
 *           <li>{@code DenseMatrix} — dense storage for small or dense systems.</li>
 *           <li>{@code CSRMatrix} / {@code PaddedCSRMatrix} — sparse storage
 *               optimized for large systems with fixed sparsity patterns.</li>
 *           <li>{@code HashSparseMatrix} — flexible sparse storage used during
 *               incremental matrix construction.</li>
 *         </ul>
 *       </li>
 *
 *       <li>
 *         {@code operators.nonlinear} —
 *         Nonlinear operators and tensor representations:
 *         <ul>
 *           <li>{@code PaddedCSR2Tensor} — sparse quadratic tensor operator.</li>
 *           <li>{@code PaddedCSRTensor} — sparse higher-order tensor operator.</li>
 *         </ul>
 *       </li>
 *
 *       <li>
 *         {@code operators.vectors} —
 *         Vector data structures used as inputs and outputs of numerical
 *         operators and solvers.
 *       </li>
 *     </ul>
 *   </dd>
 *
 *   <dt>{@link com.rae.formicapi.foundation.math.solvers}</dt>
 *   <dd>
 *     Numerical solvers for linear and nonlinear systems:
 *     <ul>
 *       <li>{@code LeastSquare} — iterative least-squares solver.</li>
 *       <li>{@code ConjugateGradient} — solver for symmetric positive-definite
 *           linear systems.</li>
 *       <li>{@code NewtonKrylov} — matrix-free nonlinear solver using
 *           Jacobian-vector products.</li>
 *     </ul>
 *   </dd>
 *
 *   <dt>{@link com.rae.formicapi.foundation.math.data}</dt>
 *   <dd>
 *     Data structures for tabulated functions and interpolation used by
 *     numerical models:
 *     <ul>
 *       <li>{@code OneDTabulatedFunction} — one-dimensional interpolation.</li>
 *       <li>{@code ReversibleOneDTabulatedFunction} — invertible lookup.</li>
 *       <li>{@code TwoDTabulatedFunction} / {@code TwoDSparseTabulatedFunction}
 *           — two-dimensional interpolation over dense and sparse grids.</li>
 *     </ul>
 *   </dd>
 * </dl>
 */
@NonnullDefault
package com.rae.formicapi.foundation.math;

import org.lwjgl.system.NonnullDefault;