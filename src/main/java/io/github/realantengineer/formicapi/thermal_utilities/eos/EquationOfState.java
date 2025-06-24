package io.github.realantengineer.formicapi.thermal_utilities.eos;

public interface EquationOfState {
    double R = 8.314462618; // [J/mol·K]
    /**
     * @param temperature temperature in Kelvin
     * @param volumeMolar molar volume in m^3/mol
     * @return pressure [Pa]
     */
    double pressure(double temperature, double volumeMolar);

    /**
     * @param temperature temperature in Kelvin
     * @param pressure pressure in Pascals
     * @param vaporFraction fraction of vapor bwn 0 and 1
     * @return molar volume in [m^3/mol]
     */
    default double volumeMolar(double temperature, double pressure, double vaporFraction) {
        throw new UnsupportedOperationException("Not implemented");
    }



}
