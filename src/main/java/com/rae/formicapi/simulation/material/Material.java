package com.rae.formicapi.simulation.material;

public enum Material {
    //todo add other properties : viscosity, solid/liquid/gas, tensile strength, meltdown temperature....
    ALUMINUM(205),
    COPPER(385),
    STEEL(50),
    WATER(0.6),
    AIR(0.025);

    private final double conductivity;

    Material(double conductivity) {
        this.conductivity = conductivity;
    }

    public double getConductivity() {
        return conductivity;
    }
}
