package com.rae.formicapi.new_thermalmodels;

/**
 * Fundamental equation of state in mass-based units (kg)
 */
public interface EquationOfState {

    double R = 8.314462618; // J/mol·K
    double T_REF = 298.15;
    double P_REF = 101325.0;

    /** Pressure [Pa] given T [K] and mass volume [m³/kg] */
    double pressure(double T, double Vm_mass);

    /** Derivatives */
    double dPdV(double T, double Vm_mass);
    double dPdT(double T, double Vm_mass);

    /** Helmholtz free energy per unit mass [J/kg] */
    default double helmholtzFreeEnergy(double T, double Vm_mass) {
        throw new UnsupportedOperationException();
    }

    /** Molar mass [kg/mol] */
    double getM();
}