package com.rae.formicapi.thermal_utilities.eos;

public interface EquationOfState {
    double R = 8.314462618; // [J/mol·K]
    /**
     * @param temperature temperature in Kelvin
     * @param volumeMolar molar volume in m^3/mol
     * @return pressure [Pa]
     */
    double pressure(double temperature, double volumeMolar);


}
