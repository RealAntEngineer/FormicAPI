package com.rae.formicapi.simulation.nodal.core;

import com.rae.formicapi.simulation.nodal.ModelType;

import java.util.EnumMap;
import java.util.Map;

/**
 * A node that is unknown and solved for in every domain it participates in.
 *
 * <p>Stores one value per domain. The capacitance map carries storage terms
 * for transient solves (thermal capacity, fluid compressibility, inertia, etc.).
 *  * @see Node
 *  * @see ModelType
 */
public class UnknownNode extends Node {

    private final Map<ModelType, Double> values       = new EnumMap<>(ModelType.class);
    private final Map<ModelType, Double> capacitances = new EnumMap<>(ModelType.class);

    /**
     * Constructs a node unknown in all given domains.
     *
     * @param capacitance map of domain → storage term (use empty map for steady-state only)
     * @param first        at least one domain required
     * @param rest         additional domains
     */
    public UnknownNode(Map<ModelType, Double> capacitance, ModelType first, ModelType... rest) {
        super(first, rest);
        for (ModelType t : getDomains()) {
            values.put(t, 0.0);
            this.capacitances.put(t, capacitance.getOrDefault(t, 0.0));
        }
    }

    /** Convenience constructor — zero capacitance in all domains. */
    public UnknownNode(ModelType first, ModelType... rest) {
        this(Map.of(), first, rest);
    }

    @Override public boolean isUnknown(ModelType type) { return true; }

    @Override
    public double getValue(ModelType type) {
        assertParticipates(type);
        return values.get(type);
    }

    @Override
    public void setValue(ModelType type, double value) {
        assertParticipates(type);
        values.put(type, value);
    }

    public double getCapacitance(ModelType type) {
        return capacitances.getOrDefault(type, 0.0);
    }

    private void assertParticipates(ModelType type) {
        if (!participatesIn(type))
            throw new IllegalArgumentException("Node does not participate in " + type);
    }
}