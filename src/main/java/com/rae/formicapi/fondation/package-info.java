/**
 * Core abstractions and building blocks for the formicapi simulation framework.
 *
 * <p>This package and its sub-packages form the foundation layer of formicapi —
 * domain-agnostic infrastructure that the rest of the library builds on.
 * Nothing here carries product-specific or application-specific knowledge.
 *
 * <h2>Sub-packages</h2>
 *
 * <dl>
 *   <dt>{@link com.rae.formicapi.fondation.simulation}</dt>
 *   <dd>Nodal network simulation engine: nodes, components, domain models,
 *       solvers, and multi-physics coupling. The central sub-package of
 *       this library.</dd>
 *
 *   <dt>{@link com.rae.formicapi.fondation.math}</dt>
 *   <dd>Mathematical primitives used by the simulation engine: sparse and
 *       dense matrix implementations, linear solvers, numerical differentiation,
 *       and tabulated function interpolation.</dd>
 *
 *   <dt>{@link com.rae.formicapi.fondation.units}</dt>
 *   <dd>Physical unit contracts and typed quantity enums (temperature,
 *       pressure, irradiation flux) to reduce raw-double mistakes at
 *       component interfaces.</dd>
 * </dl>
 *
 * <h2>Design principles</h2>
 *
 * <ul>
 *   <li><b>Domain-agnostic core.</b> The nodal engine works identically for
 *       thermal, mechanical, hydraulic, and electrical networks. Physical meaning
 *       is carried by {@link com.rae.formicapi.fondation.simulation.nodal.ModelType},
 *       not by separate class hierarchies.</li>
 *   <li><b>Separation of structure and algorithm.</b> Matrix assembly
 *       ({@code stamp}) is decoupled from solving ({@code ModelType.solve}),
 *       so new solver strategies can be introduced without touching component
 *       implementations.</li>
 *   <li><b>Additive stamping.</b> All components contribute to system matrices
 *       by addition only, allowing multiple components sharing a node to
 *       accumulate independently and correctly.</li>
 * </ul>
 */

package com.rae.formicapi.fondation;