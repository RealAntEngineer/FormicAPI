package com.rae.formicapi.thermal_utilities;


import com.rae.formicapi.FormicAPI;
import com.rae.formicapi.math.Solvers;
import net.createmod.catnip.data.Couple;
import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

public class PengRobinsonEOS implements EquationOfState {
    public static final double R = 8.314462618; // [J/mol·K]
    private final double Tc;  // Critical temperature [K]
    private final double Pc;  // Critical pressure [Pa]
    private final double omega; // Acentric factor
    private final double kappa;
    private final double a0;
    private final double b;

    //TODO add molar weight to transform molar
    public PengRobinsonEOS(double Tc, double Pc, double omega) {
        this.Tc = Tc;
        this.Pc = Pc;
        this.omega = omega;
        this.kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega * omega;
        this.a0 = 0.45724 * R * R * Tc * Tc / Pc;
        this.b = 0.07780 * R * Tc / Pc;

    }

    public double alpha(double T) {
        double Tr = T / Tc;
        return Math.pow(1 + kappa * (1 - Math.sqrt(Tr)), 2);
    }

    private double dAlpha_dT(double T) {
        double sqrtTr = Math.sqrt(T / Tc);
        return -kappa * (1 + kappa * (1 - sqrtTr)) / (Tc * sqrtTr);
    }

    private double da_dT(double T) {
        return a0 * dAlpha_dT(T); // ac is the "a" at critical temperature
    }

    public double a(double T) {
        return a0 * alpha(T);
    }

    @Override
    public double pressure(double T, double Vm) {
        double a = a(T);
        double term1 = R * T / (Vm - b);
        double term2 = a / (Vm * (Vm + b) + b * (Vm - b));
        return term1 - term2;
    }
    @Override
    public double volumeMolar(double T, double P, double vaporFraction) {
        double[] roots = getZFactors(T, P);

        // Filter for real positive roots
        double[] realRoots = Arrays.stream(roots)

                .toArray();

        if (realRoots.length == 0) {
            throw new IllegalStateException("No valid real roots for Z.");
        }

        double Z;
        if (vaporFraction <= 0.0) {
            Z = realRoots[0]; // liquid root (smallest Z)
        } else if (vaporFraction >= 1.0) {
            Z = realRoots[realRoots.length - 1]; // vapor root (largest Z)
        } else if (realRoots.length >= 2) {
            double Zl = realRoots[0];
            double Zv = realRoots[realRoots.length - 1];
            Z = (1 - vaporFraction) * Zl + vaporFraction * Zv;
        } else {
            // Only one root, fallback
            Z = realRoots[0];
        }

        return Z * R * T / P;
    }

    /**
     *  What does this represent exactly ?
     * @param T the temperature in Kelvin, needs to be strictly positive
     * @param P the pressure in Pascals, needs to be strictly positive
     * @return the possible compressibility coefficient for the given isotherm and Pressure : minimal is the value at the saturation curve for liquid
     * the middle is non-physical, and the max is for the vapor
     */
    public double @NotNull [] getZFactors(double T, double P) {
        assert T > 0;
        assert P > 0;
        double A = a(T) * P / (R * R * T * T);
        double B = b * P / (R * T);

        // Coefficients of the cubic in Z
        double c1 = 1.0;
        double c2 = -(1.0 - B);
        double c3 = A - 3 * B * B - 2 * B;
        double c4 = -(A * B - B * B - B * B * B);

        // Solve cubic
        return Arrays.stream(Solvers.solveCubic(c1, c2, c3, c4))
                .filter(d -> !Double.isNaN(d) && d > minZ((float) T, (float) P))
                .sorted()
                .toArray();
    }
    //not robust at low pressure
    public double fugacityCoefficient(double T, double P, double Z) {
        double R = PengRobinsonEOS.R;
        double a = a(T); // temperature-dependent 'a'
        double A = a * P / (R * R * T * T);
        double B = b * P / (R * T);
        double sqrt2 = Math.sqrt(2);

        double term1 = Z - 1 - Math.log(Z - B);
        double term2 = A / (2 * sqrt2 * B);
        double logArgument = (Z + (1 + sqrt2) * B) / (Z + (1 - sqrt2) * B);
        double term3 = term2 * Math.log(logArgument);

        double lnPhi = term1 - term3;

        System.out.println("fug : "+Math.exp(lnPhi));
        return Math.exp(lnPhi);
    }

    public List<Double> findSpinodalPoints(double T, double vMin, double vMax) {
        List<Double> spinodals = new ArrayList<>();

        double previousV = vMin;
        double previousP = pressure(T, previousV);
        double previousSlope = Double.NaN;

        for (double V = vMin * 1.01; V <= vMax; V *= 1.01) {
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

    public double computeSaturationPressure(double T) {
        if (T > Tc) {
            throw new IllegalArgumentException("T must be lower than the critical temperature to compute a coexistence pressure");
        }
        if (T <= 0) {
            throw new IllegalArgumentException("Temperature must be positive");
        }

        // Estimate pressure range based on spinodal points
        List<Double> spinodals = findSpinodalPoints(T, b * 1.001f, 1);
        if (spinodals.isEmpty()) {
            throw new RuntimeException("No spinodal points found");
        }
        assert spinodals.size() == 2;

        double pMin = Math.max(1e-3, pressure(T, spinodals.get(0)));  // lower pressure bound
        double pMax = pressure(T, spinodals.get(1));               // upper pressure bound

        // Fugacity difference function
        Function<Float, Float> areaDifferenceFunction = (Float Guess) -> {
            double P = Guess;
            double[] roots = getZFactors(T, P);
            if (roots.length < 2) return Float.MAX_VALUE;

            double Zl = Arrays.stream(roots).min().getAsDouble();
            double Zv = Arrays.stream(roots).max().getAsDouble();
            double Vl = Zl * PengRobinsonEOS.R * T / P;
            double Vv = Zv * PengRobinsonEOS.R * T / P;
            if (Math.abs(Vv - Vl) < 1e-9) return Float.MAX_VALUE;

            int n = 1000;
            double dV = (Vv - Vl) / n;
            double area = 0;
            for (int i = 0; i < n; i++) {
                double V1 = Vl + i * dV;
                double V2 = V1 + dV;
                double p1 = pressure(T, V1);
                double p2 = pressure(T, V2);
                area += 0.5 * (p1 + p2) * dV;
            }

            double rect = P * (Vv - Vl);
            return (float)Math.abs(area - rect);
        };
        Function<Float, Float> fugacityDifference = (Float P) -> {
            try {
                double[] Zroots = getZFactors(T, P);
                if (Zroots.length < 2) return Float.MAX_VALUE;

                double Zl = Arrays.stream(Zroots).min().getAsDouble();
                double Zv = Arrays.stream(Zroots).max().getAsDouble();

                double phiL = fugacityCoefficient(Zl, T, P);
                double phiV = fugacityCoefficient(Zv, T, P);
                return (float) Math.abs(Math.log(phiV) - Math.log(phiL));
            } catch (Exception e) {
                return Float.MAX_VALUE;
            }
        };

        return Solvers.gradientDecent(areaDifferenceFunction, (float)Math.max(1e-3,(pMin+pMax)/2), 10,1e-3f,1e-4f);
    }

    /**
     * Waarning only use this if you know that the fuild is in the LV phase.
     * @param T temperature
     * @return return the couple Vl, Vv or the intermediate V if it's super critical.
     */
    @Deprecated
    public Couple<Double> getSaturationVolumes(double T) {
        try {
            if (T < Tc ) {

                double PSat = computeSaturationPressure(T); // try normal coexistence
                double Vl = volumeMolar(T, PSat, 0.0); // saturated liquid
                double Vv = volumeMolar(T, PSat, 1.0); // saturated vapor
                return Couple.create(Vl, Vv);
            } else {
                double[] critRoot = getZFactors(Tc, Pc);

                // Start guess: approximate ideal gas volume
                double Z = Arrays.stream(critRoot).max().getAsDouble();
                //we need to find a better interpolation function.
                double VmInflection = Z * R * Tc / Pc;
                return Couple.create(VmInflection, VmInflection);
            }
        } catch (RuntimeException e) {
            FormicAPI.LOGGER.debug("Unexpectedly found outside LV phase {}", T);
            FormicAPI.LOGGER.debug(e);
            // Fall back to inflection point
            return Couple.create(Double.NaN, Double.NaN);
        }
    }

    public double idealGasEntropy(double T, double P) {
        double Cp = 3.5 * R;
        double Pref = 101325.0;
        double Tref = 298.15;

        return Cp * Math.log(T / Tref) - R * Math.log(P / Pref);
    }

    public double entropyDeparturePR(double T, double Vm) {
        double a = a(T);  // temperature-dependent 'a' from PR EOS
        double sqrt2 = Math.sqrt(2);

        double term1 = R * Math.log((Vm - b) / Vm);
        double term2 = - (a / (2 * sqrt2 * b * R * T)) *
                Math.log((Vm + (1 + sqrt2) * b) / (Vm + (1 - sqrt2) * b));

        return term1 + term2;
    }
    public double totalEntropy(double T, double Vm) {
        // Ideal gas entropy (reference + integration)
        double sIdeal = idealGasEntropy(T, pressure(T, Vm));
        // Entropy departure from Peng-Robinson EOS
        double sDeparture = entropyDeparturePR(T, Vm);

        return sIdeal + sDeparture;
    }


    /**
     *
     * @param T temperature
     * @return molar enthalpy
     */
    private double idealGasEnthalpy(double T) {
        double Cp = 75.0; // J/mol·K
        return Cp * (T - 273.15);
    }

    /**
     *
     * @param T temperature
     * @param Vm specific volume
     * @return the residual enthalpy (H^res) at given temperature and molar volume.
     * Units: J/mol
     */
    private double residualEnthalpy(double T, double Vm) {
        double a = a(T); // attraction parameter at T
        double da_dT = da_dT(T); // temperature derivative of a
        double P = pressure(T, Vm);
        double Z = P * Vm / (R * T);

        // PR-specific constants
        double sqrt2 = Math.sqrt(2);
        double B = b;
        double logTerm = Math.log((Vm + (1 + sqrt2) * B) / (Vm + (1 - sqrt2) * B));

        // Correct residual enthalpy formula
        double hRes = R * T * (Z - 1) + (T * da_dT - a) / (B * sqrt2) * logTerm;

        return hRes;
    }


    public double totalEnthalpy(double T, double Vm) {
        return idealGasEnthalpy(Math.max(T,1)) + residualEnthalpy(Math.max(T,1), Math.max(Vm,b));
    }

    public double getB() {
        return b;
    }//this should be the starting point for plotting

    public double minZ(float T, float P) {
        return (b *1.001f) * P / (R * T); // Minimum physically valid Z
    }
}

