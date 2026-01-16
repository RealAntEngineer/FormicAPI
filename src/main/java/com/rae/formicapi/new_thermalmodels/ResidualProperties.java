package com.rae.formicapi.new_thermalmodels;

public interface ResidualProperties {

    double residualEntropy(double T, double Vm_mass);

    double totalEntropy(double T, double Vm_mass);

    double residualEnthalpy(double T, double Vm_mass);

    double totalEnthalpy(double T, double Vm_mass);


}
