/**
 * Nodal network simulation engine supporting multiple physical domains.
 *
 * <p>A nodal simulation models a physical system as a graph of
 * {@link com.rae.formicapi.fondation.simulation.nodal.core.Node nodes} connected
 * by {@link com.rae.formicapi.fondation.simulation.nodal.core.SimulationComponent components}.
 * Each domain obeys the same conductance relation:
 *
 * <pre>
 *     flow = conductance × (potential_a − potential_b)
 * </pre>
 *
 * <p>The physical meaning of potential and flow changes with the domain:
 *
 * <table border="1" style="border-collapse:collapse; text-align:left">
 *   <tr><th>Domain</th><th>Potential</th><th>Flow</th></tr>
 *   <tr><td>THERMAL</td>   <td>Temperature [K]</td>        <td>Heat flux [W]</td></tr>
 *   <tr><td>MECHANICAL</td><td>Angular velocity [rad/s]</td><td>Torque [N·m]</td></tr>
 *   <tr><td>HYDRAULIC</td> <td>Pressure [Pa]</td>           <td>Mass flow [kg/s]</td></tr>
 * </table>
 *
 * <h2>Sub-packages</h2>
 *
 * <dl>
 *   <dt>{@link com.rae.formicapi.fondation.simulation.nodal.core}</dt>
 *   <dd>Fundamental types: {@code Node}, {@code SimulationComponent},
 *       {@code DomainModel}, {@code SimulationModel}, {@code SimulationContext},
 *       and {@code LinearLink}. Start here when building a new simulation.</dd>
 *
 *   <dt>{@link com.rae.formicapi.fondation.simulation.nodal.linear}</dt>
 *   <dd>Ready-made single-domain components for thermal, mechanical, and
 *       hydraulic networks (convection, radiation, rotational dampers, gears).</dd>
 *
 *   <dt>{@link com.rae.formicapi.fondation.simulation.nodal.complex}</dt>
 *   <dd>Multi-domain components that couple two or more physical domains,
 *       such as heat exchangers and cooling jackets.</dd>
 * </dl>
 *
 * <h2>Solving</h2>
 *
 * <p>{@link com.rae.formicapi.fondation.simulation.nodal.SteadyStateSolver}
 * orchestrates the outer fixed-point loop that converges cross-domain coupling.
 * Each domain's inner solve strategy is owned by its
 * {@link com.rae.formicapi.fondation.simulation.nodal.ModelType} — linear domains
 * use a least-squares solver; the mechanical domain uses Newton-Raphson to handle
 * quadratic losses correctly.
 */
package com.rae.formicapi.fondation.simulation.nodal;