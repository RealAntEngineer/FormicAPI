package com.rae.formicapi.content.thermal_utilities;


/**
 * main definition is Pressure, enthalpy, could also be P, S
 */
public interface EOS {

    float getS(float P, float H);

    float getH(float P, float S);

    float getT(float P, float H);

    float density(float P, float H);

    Phase.PhaseMix getPhases();
}
