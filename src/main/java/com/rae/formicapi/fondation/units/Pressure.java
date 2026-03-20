package com.rae.formicapi.fondation.units;

public enum Pressure implements IUnit {
    PASCALS(1),
    BAR(1f / 100000),
    ATMOSPHERES(1f / 101300),
    PSI(1f / 6895);

    private final float a;

    Pressure(float a) {
        this.a = a;
    }

    public float convert(float pascal) {
        return pascal * a;
    }

}
