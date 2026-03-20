package com.rae.formicapi.simulation.nodal.core;

import com.rae.formicapi.simulation.nodal.ModelType;

import java.util.EnumMap;
import java.util.EnumSet;
import java.util.Map;
import java.util.Set;

/**
 * Abstract base for all nodes in a nodal network simulation.
 *
 * <p>A node represents a physical point in the network that may participate
 * in one or more physical domains simultaneously. For example, a fluid node
 * carries both a pressure (HYDRAULIC) and a temperature (THERMAL).
 *
 * <p>Each domain the node participates in gets an independent matrix index,
 * assigned per domain by {@link DomainModel#rebuildContext()}.
 *
 * @see UnknownNode
 * @see FixedValueNode
 * @see ModelType
 */
public abstract class Node {

    private final Set<ModelType> domains;
    private final Map<ModelType, Integer> ids = new EnumMap<>(ModelType.class);

    protected Node(ModelType first, ModelType... rest) {
        this.domains = rest.length == 0
                ? EnumSet.of(first)
                : EnumSet.of(first, rest);
    }

    // ── domain membership ─────────────────────────────────────────────────

    /** All domains this node participates in. */
    public Set<ModelType> getDomains() { return domains; }

    public boolean participatesIn(ModelType type) { return domains.contains(type); }

    // ── matrix index — one per domain ─────────────────────────────────────

    /**
     * Returns the matrix row/column index for this node in the given domain.
     *
     * @throws IllegalStateException if the domain has not been registered yet
     * @throws IllegalArgumentException if this node does not participate in that domain
     */
    public int getId(ModelType type) {
        if (!domains.contains(type))
            throw new IllegalArgumentException(
                    "Node does not participate in domain " + type);
        Integer id = ids.get(type);
        if (id == null)
            throw new IllegalStateException(
                    "getId(" + type + ") called before setId() for this domain");
        return id;
    }

    /** Called by {@link DomainModel#rebuildContext()} once per domain per rebuild. */
    public void setId(ModelType type, int id) {
        if (!domains.contains(type))
            throw new IllegalArgumentException(
                    "Cannot assign id for domain " + type + " — node does not participate");
        ids.put(type, id);
    }

    // ── per-domain value contract ─────────────────────────────────────────

    /**
     * Whether this node is an unknown (to be solved) in the given domain.
     * A node may be fixed in one domain and unknown in another.
     */
    public abstract boolean isUnknown(ModelType type);

    /** Returns the current scalar value for the given domain. */
    public abstract double getValue(ModelType type);

    /** Sets the value for the given domain. Fixed domains silently ignore this. */
    public abstract void setValue(ModelType type, double value);
}