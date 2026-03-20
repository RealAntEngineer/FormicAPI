package com.rae.formicapi.fondation.units;

public enum Temperature implements IUnit {
    KELVIN(1, 0),
    CELSIUS(1, -273.15f),
    FAHRENHEIT(9f / 5, -273.15f * 9f / 5 + 32);

    private final float a;
    private final float b;

    Temperature(float a, float b) {
        this.a = a;
        this.b = b;
    }

    public float convert(float kelvin) {
        return kelvin * a + b;
    }

}
