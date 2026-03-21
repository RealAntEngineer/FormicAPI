package com.rae.formicapi.fondation.simulation.nodal.core;

import com.rae.formicapi.fondation.simulation.nodal.ModelType;

import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * A physical component that participates in one or more simulation domains
 * by contributing terms to the nodal system matrices.
 *
 * <p>Components are the building blocks of a nodal network. They fall into
 * two broad categories:
 *
 * <ul>
 *   <li><b>Single-domain links</b> — connect two nodes within the same physical
 *       domain via a conductance-like relation (e.g. {@code LinearLink},
 *       {@code RotationalDamper}). These stamp only one {@link SimulationContext}.</li>
 *   <li><b>Multi-domain components</b> — couple distinct physical domains, converting
 *       a quantity in one domain into a source term in another (e.g. a
 *       {@code RotationalDamper} converting mechanical dissipation into a thermal
 *       heat source, or a heat exchanger coupling a fluid and a solid domain).</li>
 * </ul>
 *
 * <h2>Node ownership</h2>
 *
 * <p>A component's nodes are split into two roles:
 *
 * <ul>
 *   <li><b>Interface nodes</b> ({@link #getInterfaceNodes()}) — nodes that must
 *       already exist in the network and be connected to other components for the
 *       simulation to be physically meaningful. The caller is responsible for
 *       providing and wiring these.</li>
 *   <li><b>Internal nodes</b> ({@link #getInternalNodes()}) — nodes that the
 *       component creates and owns, invisible to the caller. Typical uses are
 *       Lagrange multiplier rows for kinematic constraints (e.g. gear ratios) or
 *       intermediate coupling nodes. {@link SimulationModel#addComponent} registers
 *       these automatically; callers never interact with them directly.</li>
 * </ul>
 *
 * <h2>Stamping</h2>
 *
 * <p>{@link #stamp(Map)} is called once per solver iteration after all
 * {@link SimulationContext} matrices have been freshly zeroed by
 * {@link DomainModel#rebuildContext()}. Implementations must be <em>additive</em>:
 * they add their contribution to the existing matrix and RHS entries rather than
 * overwriting them, so that multiple components sharing a node accumulate correctly.
 *
 * <p>Components that read values from a foreign domain (e.g. reading mechanical
 * speeds to compute a thermal heat source) must use the node values from the
 * <em>previous</em> iteration — the values frozen by the outer coupling loop.
 * They must not attempt to solve or update nodes in the foreign domain.
 *
 * <h2>Implementing a new component</h2>
 * <ol>
 *   <li>Declare which domains are touched via {@link #getDomains()}.</li>
 *   <li>Return any privately owned nodes from {@link #getInternalNodes()} —
 *       these are registered by the model, not the caller.</li>
 *   <li>Return all externally provided nodes from {@link #getInterfaceNodes()} —
 *       these document the wiring contract for users of the component.</li>
 *   <li>Implement {@link #stamp(Map)}, guarding each context with a null-check
 *       since a domain may be absent from the model.</li>
 * </ol>
 *
 * @see SingleDomainComponent
 * @see SimulationModel
 * @see DomainModel
 */
public interface SimulationComponent {

    /**
     * Returns the set of physical domains this component stamps into.
     *
     * <p>Used by the solver to route components to the correct
     * {@link SimulationContext} instances and to filter components
     * when solving a single domain in isolation.
     *
     * @return a non-empty, unmodifiable set of {@link ModelType} values
     */
    Set<ModelType> getDomains();

    /**
     * Returns nodes that this component creates and owns internally.
     *
     * <p>Internal nodes are an implementation detail — callers never reference
     * them. Typical examples are Lagrange multiplier rows introduced by
     * constraint components such as gears or rigid couplings.
     *
     * <p>{@link SimulationModel#addComponent} automatically registers these
     * nodes into the appropriate {@link DomainModel} when the component is added.
     *
     * @return an unmodifiable list of internally owned nodes;
     *         empty by default for components that require no extra rows
     */
    default List<Node> getInternalNodes() {
        return List.of();
    }

    /**
     * Returns nodes that must be externally provided and wired into the network.
     *
     * <p>These are the nodes the caller passes in at construction time — shaft
     * nodes, boundary temperatures, heat sink nodes, and so on. The component
     * is not valid unless all interface nodes belong to an active
     * {@link DomainModel} and are connected to at least one other component.
     *
     * <p>This list documents the physical wiring contract of the component
     * and may be used by validation or visualisation tooling to verify network
     * connectivity before solving.
     *
     * @return an unmodifiable list of externally owned interface nodes;
     *         empty by default
     */
    default List<Node> getInterfaceNodes() {
        return List.of();
    }

    /**
     * Stamps this component's contribution into the relevant simulation contexts.
     *
     * <p>Called once per outer solver iteration, after all contexts have been
     * zeroed by {@link DomainModel#rebuildContext()}. Implementations must only
     * <em>add</em> to matrix and RHS entries — never set or clear them.
     *
     * <p>The provided map contains one entry per active domain. Components that
     * span multiple domains should null-check each context they access, since a
     * domain may be absent if the model contains no nodes of that type.
     *
     * @param contexts a map from {@link ModelType} to the freshly rebuilt
     *                 {@link SimulationContext} for that domain; never null,
     *                 but may not contain an entry for every {@link ModelType}
     */
    void stamp(Map<ModelType, SimulationContext> contexts);
}