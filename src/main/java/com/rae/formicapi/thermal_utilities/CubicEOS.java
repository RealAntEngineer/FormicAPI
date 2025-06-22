package com.rae.formicapi.thermal_utilities;

import com.rae.formicapi.FormicAPI;
import net.createmod.catnip.data.Couple;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public abstract class CubicEOS implements EquationOfState{

    public static final double T_REF = 298.15;
    public static final double P_REF = 101325.0;
    protected final double M;
    protected final double Tc;  // Critical temperature [K]
    protected final double Pc;  // Critical pressure [Pa]


    protected CubicEOS(double m, double Tc, double Pc) {
        this.M = m;
        this.Tc = Tc;
        this.Pc = Pc;
    }

    @Override
    public abstract double pressure(double temperature, double volumeMolar);


    public abstract List<Double> findSpinodalPoints(double T);
    public List<Double> findSpinodalPoints(double T, double vMin, double vMax) {
        List<Double> spinodals = new ArrayList<>();

        double previousV = vMin;
        double previousP = pressure(T, previousV);
        double previousSlope = Double.NaN;

        for (double V = vMin; V <= vMax; V *= 1.05) {
            double P = pressure(T, V);
            double slope = (P - previousP) / (V - previousV);

            if (!Double.isNaN(previousSlope) && previousSlope * slope <= 0) {
                // slope changed sign → inflection/spinodal
                spinodals.add(V);
            }

            previousV = V;
            previousP = P;
            previousSlope = slope;
        }

        return spinodals;
    }

    @Override
    public double volumeMolar(double T, double P, double vaporFraction) {
        double[] roots = getZFactors(T, P);

        if (roots.length == 0) {
            throw new IllegalStateException("No valid real roots for Z at T = "+ T+ " P = "+ P);
        }

        double Z;
        if (vaporFraction <= 0.0) {
            Z = roots[0]; // Liquid-like root
        } else if (vaporFraction >= 1.0) {
            Z = roots[roots.length - 1]; // Vapor-like root
        } else if (roots.length >= 2) {
            double saturationPressure = saturationPressure(T);
            // Interpolate molar volume, not Z
            double Zl = roots[0];
            double Zv = roots[roots.length - 1];
            double Vl = Zl * R * T / saturationPressure;
            double Vv = Zv * R * T / saturationPressure;
            return (1 - vaporFraction) * Vl + vaporFraction * Vv;
        } else {
            // Fallback for single phase
            Z = roots[0];
        }

        return Z * R * T / P;
    }

    abstract double[] getZFactors(double T, double P);
    protected double idealGasEntropy(double T, double P) {
        if (T < 0) T = 0.1f;
        if (P < 0) P = 0.1f;

        double Cp = 3.5 * R;

        return (Cp * Math.log(T) - R * Math.log(P))/M;
    }
    abstract double saturationPressure(double T);

    /**
     * @param T temperature
     * @param Vm molar volume -> TODO go toward a by mass volume
     * @return specific entropy
     */
    public final double totalEntropy(double T, double Vm) {
        double SRef = idealGasEntropy(T_REF, P_REF) + residualEntropy(T_REF, P_REF,  getZFactors(T_REF, P_REF)[0]);

        // Check two-phase region
        if (T >= Tc ) {//we can't compute the saturation pressure past the critical temperature by definition
            // Not in 2-phase region, just compute entropy at T, P, Z
            double P = pressure(T, Vm); // From PR EOS
            double Z = Vm * P / (T * R);//only one roots
            return residualEntropy(T, P, Z) + idealGasEntropy(T,P) - SRef;
        }
        double PSat = saturationPressure(T);//if we are 2 phased then we need to use the saturation pressure

        // Two-phase region
        double[] Zs = getZFactors(T, PSat);
        if (Zs.length == 0) {//if there is no solution we are a liquid that is too low in temperature.
            double P = pressure(T, Vm); // From PR EOS
            double Z = Vm * P / (T * R);//only one roots
            return residualEntropy(T, P, Z) + idealGasEntropy(T,P)- SRef;
        }
        double Z_l = Zs[0];
        double Z_v = Zs[Zs.length - 1];

        // Compute molar volumes of saturated phases
        double v_l = Z_l * R * T / PSat;
        double v_v = Z_v * R * T / PSat;

        if (Vm < v_l || Vm > v_v){//check if 2 phases or not
            double P = pressure(T, Vm); // From PR EOS
            double Z = Vm * P / (T * R);//only one roots
            return residualEntropy(T, P, Z) + idealGasEntropy(T,P)- SRef;
        }

        // Compute vapor quality x
        double x = (Vm - v_l) / (v_v - v_l);
        x = Math.max(0.0, Math.min(1.0, x)); // Clamp between 0 and 1

        // Residual entropy for each phase
        double S_l = residualEntropy(T, pressure(T, v_l), Z_l); //+ idealGasEntropy(T, PSat);
        double S_v = residualEntropy(T,  pressure(T, v_v), Z_v);// + idealGasEntropy(T, PSat);
        // Mix and convert to specific entropy
        return (1 - x) * S_l + x * S_v - SRef + idealGasEntropy(T, PSat); // [J/kg·K]
    }

    /**
     * Warning only use this if you know that the fluid is in the LV phase.
     * @param T temperature
     * @return return the couple Vl, Vv or the intermediate V if it's super critical.
     */
    public Couple<Double> getSaturationVolumes(double T) {
        try {
            if (T < Tc ) {
                double PSat = saturationPressure(T); // try normal coexistence
                double[] roots = getZFactors(T, PSat);

                if (roots.length <= 2) {
                    throw new IllegalStateException("The Saturation Pressure is not in a 2 phase region");
                }
                double Zl = roots[0];
                double Zv = roots[roots.length - 1];
                double Vl = Zl * R * T / PSat;
                double Vv = Zv * R * T / PSat;
                return Couple.create(Vl, Vv);
            } else {
                double[] critRoot = getZFactors(Tc, Pc);

                // Start guess: approximate ideal gas volume
                double Z = Arrays.stream(critRoot).max().getAsDouble();
                //we need to find a better interpolation function.
                double VmInflection = Z * EquationOfState.R * Tc / Pc;
                return Couple.create(VmInflection, VmInflection);
            }
        } catch (RuntimeException e) {
            FormicAPI.LOGGER.debug("Unexpectedly found outside LV phase {}", T);
            FormicAPI.LOGGER.debug(e);
            // Fall back to inflection point
            return Couple.create(Double.NaN, Double.NaN);
        }
    }

    protected abstract double residualEntropy(double T, double P, double Z);


    /**
     *
     * @param T temperature
     * @return specific enthalpy
     */
    protected double idealGasEnthalpy(double T) {
        double Cp = 75.0; // J/mol·K
        return Cp * (T)/M;
    }

    protected abstract double residualEnthalpy(double T, double P, double Z);

    public double totalEnthalpy(double T, double Vm) {
        double HRef = idealGasEnthalpy(T_REF) + residualEnthalpy(T_REF, P_REF,  getZFactors(T_REF, P_REF)[0]);
        double P = pressure(T, Vm); // From PR EOS

        // Check two-phase region
        if (T >= Tc ) {//we can't compute the saturation pressure past the critical temperature by definition
            // Not in 2-phase region, just compute entropy at T, P, Z
            double Z = Vm * P / (T * R);//only one roots
            return residualEnthalpy(T, P, Z) + idealGasEnthalpy(T) - HRef;
        }

        double PSat = saturationPressure(T);//if we are 2 phased then we need to use the saturation pressure

        // Two-phase region
        double[] Zs = getZFactors(T, PSat);

        if (Zs.length == 0){//check if 2 phases or not
            System.out.println("weird T "+T);
            double Z = Vm * P / (T * R);//only one roots
            return residualEnthalpy(T, P, Z) + idealGasEnthalpy(T) - HRef;
        }

        double Z_l = Zs[0];
        double Z_v = Zs[Zs.length - 1];

        // Compute molar volumes of saturated phases
        double v_l = Z_l * R * T / PSat;
        double v_v = Z_v * R * T / PSat;

        if (Vm < v_l || Vm > v_v){//check if 2 phases or not
            double Z = Vm * P / (T * R);//only one roots
            return residualEnthalpy(T, P, Z) + idealGasEnthalpy(T) - HRef;
        }


        // Compute vapor quality x
        double x = (Vm - v_l) / (v_v - v_l);
        x = Math.max(0.0, Math.min(1.0, x)); // Clamp between 0 and 1

        // Residual entropy for each phase
        double H_l = residualEnthalpy(T, PSat, Z_l) + idealGasEnthalpy(T);
        double H_v = residualEnthalpy(T, PSat, Z_v) + idealGasEnthalpy(T);
        // Mix and convert to specific entropy
        return (1 - x) * H_l + x * H_v - HRef; // [J/kg·K]
    }

    public boolean is2phases(double T, double P) {
        double saturationPressure = saturationPressure(T);
        return P >= saturationPressure * 0.99f || P <= saturationPressure * 1.01f;// do this to avoid problem with numerical error
    }

    public double getEntropy(float T, float P, float x) {
        double[] roots1 = getZFactors(T, P);
        if (roots1.length == 0)
            return 0;//this is bad -> do it differently
        if (roots1.length == 1) {// this is when it's monophasique or past critical
            double Z1 = roots1[0];
            double Vm1 = Z1 * R * T / P;
            return totalEntropy(T, Vm1);
        }
        else {
            if (T < Tc ){
                double saturationPressure = saturationPressure(T);
                if (saturationPressure * 1.01 < P) {
                    return totalEntropy(T,roots1[0]* R * T / P);
                }
                if (saturationPressure * 0.99 > P) {
                    return totalEntropy(T,roots1[roots1.length-1]* R * T / P);
                }
            }
            Couple<Double> Vms = getSaturationVolumes(T);
            return  (totalEntropy(T,Vms.getFirst()) * (1- x) +totalEntropy(T, Vms.getSecond())* x);
        }
    }

    public double getEnthalpy(float T, float P, float x){
        final double R = EquationOfState.R;
        if (Float.isNaN(T)) T = 100;
        if (Float.isNaN(P)) P = 1;

        if (T < 100) T = 100;
        if (P < 1) P = 1;
        return totalEnthalpy(T, volumeMolar(T, P, x));
    }

    protected abstract double minZ(float finalT, float finalP);
}
