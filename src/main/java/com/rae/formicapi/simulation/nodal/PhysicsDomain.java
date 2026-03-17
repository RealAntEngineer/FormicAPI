package com.rae.formicapi.simulation.nodal;

/**
 * Enumerates the supported physical domains in the nodal network.
 *
 * <p>Each domain defines the physical meaning of the node value and its
 * associated flow quantity, following the general relation:
 *
 * <pre>
 *     flow = conductance × (value_a - value_b)
 * </pre>
 *
 * <p>Useful for logging, debugging, and domain-aware printing.
 */
public enum PhysicsDomain {

    THERMAL  ("Temperature [K]",      "Heat flux [W]"      ),
    HYDRAULIC("Pressure [Pa]",        "Mass flow [kg/s]"   ),
    ELECTRICAL("Voltage [V]",         "Current [A]"        ),
    MECHANICAL("Angular velocity [rad/s]", "Torque [N·m]"  );

    public final String valueName;
    public final String flowName;

    PhysicsDomain(String valueName, String flowName) {
        this.valueName = valueName;
        this.flowName  = flowName;
    }
}