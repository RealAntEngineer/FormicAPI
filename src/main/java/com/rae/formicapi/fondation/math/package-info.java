/**
 * Mathematical primitives used by the formicapi simulation engine.
 *
 * <p>This package provides the numerical infrastructure that underpins
 * matrix assembly and solving across all physical domains. It is intentionally
 * decoupled from simulation concepts — nothing here references nodes,
 * components, or physical quantities.
 *
 * <h2>Sub-packages</h2>
 *
 * <dl>
 *   <dt>{@link com.rae.formicapi.fondation.math.operators}</dt>
 *   <dd>Matrix abstractions and implementations:
 *     <ul>
 *       <li>{@code Matrix} / {@code MutableMatrix} — read and read-write contracts.</li>
 *       <li>{@code DenseMatrix} — full storage, suitable for small systems.</li>
 *       <li>{@code DynamicCSRMatrix} / {@code CSRMatrix} — sparse storage for
 *           large nodal networks with few non-zero entries per row.</li>
 *       <li>{@code HashSparseMatrix} — insertion-order-friendly sparse matrix
 *           used during incremental assembly before conversion to CSR.</li>
 *     </ul>
 *   </dd>
 *
 *   <dt>{@link com.rae.formicapi.fondation.math.solvers}</dt>
 *   <dd>Linear system solvers:
 *     <ul>
 *       <li>{@code LeastSquare} — iterative least-squares solver, default for
 *           thermal and hydraulic domains.</li>
 *       <li>{@code ConjugateGradient} — CG solver for symmetric positive-definite
 *           systems.</li>
 *     </ul>
 *   </dd>
 *
 *   <dt>{@link com.rae.formicapi.fondation.math.data}</dt>
 *   <dd>Tabulated function interpolation for material properties and boundary
 *       conditions that vary with temperature, pressure, or other state variables:
 *     <ul>
 *       <li>{@code OneDTabulatedFunction} — 1D lookup with configurable step mode.</li>
 *       <li>{@code ReversibleOneDTabulatedFunction} — invertible 1D lookup.</li>
 *       <li>{@code TwoDTabulatedFunction} / {@code TwoDSparseTabulatedFunction}
 *           — 2D interpolation over dense and sparse grids.</li>
 *     </ul>
 *   </dd>
 * </dl>
 */
package com.rae.formicapi.fondation.math;
 