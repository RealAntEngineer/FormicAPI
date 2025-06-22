package com.rae.formicapi.thermal_utilities;


import com.rae.formicapi.math.Solvers;
import org.jetbrains.annotations.NotNull;

import java.util.*;
import java.util.function.Function;

public class PengRobinsonEOS extends CubicEOS{

    private final double omega; // Acentric factor
    private final double kappa;
    private final double a0;
    private final double b;
    private final TreeMap<Float, Float> saturationTemperature = new TreeMap<>();
    private final TreeMap<Float, Float> saturationPressure = new TreeMap<>(); // Sorted by T

    private final float PressureLogStep = 0.01f;
    private final float TemperatureStep = 0.05f;
    //TODO convert everything to float
    //TODO add molar weight to transform molar
    public PengRobinsonEOS(double Tc, double Pc, double omega, double M) {
        super(M, Tc,Pc);
        this.omega = omega;
        this.kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega * omega;
        this.a0 = 0.45724 * EquationOfState.R * EquationOfState.R * Tc * Tc / Pc;
        this.b = 0.07780 * EquationOfState.R * Tc / Pc;


        //make some precomputed things

        // Step 1: Compute (T, P) pairs
        for (float T = 0; T < Tc; T+=TemperatureStep) {
            try {
                float P = (float)computeSaturationPressure(T);
                saturationPressure.put((float) T, P);
            } catch (Exception ignored) {
            }
        }

        // Step 2: Build regular grid in log(P) space
        List<Map.Entry<Float, Float>> entries = new ArrayList<>(saturationPressure.entrySet());

        // Determine the log pressure range
        float logP_start = (float) Math.log(entries.get(0).getValue());
        float logP_end = (float) Math.log(entries.get(entries.size() - 1).getValue());
        int numSteps = (int) ((logP_end - logP_start) / PressureLogStep);


        for (int i = 0; i < numSteps; i++) {
            float targetLogP = logP_start + i * PressureLogStep;

            // Find where targetLogP fits between logP1 and logP2
            for (int j = 0; j < entries.size() - 1; j++) {
                float T1 = entries.get(j).getKey();
                float T2 = entries.get(j + 1).getKey();
                float logP1 = (float) Math.log(entries.get(j).getValue());
                float logP2 = (float) Math.log(entries.get(j + 1).getValue());

                if (targetLogP >= logP1 && targetLogP <= logP2 && logP1 != logP2) {
                    // Linear interpolation in logP
                    float t = (targetLogP - logP1) / (logP2 - logP1);
                    float T_interp = T1 + t * (T2 - T1);

                    saturationTemperature.put((float) Math.exp(targetLogP), T_interp);
                    break;
                }
            }
        }
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
        double term1 = EquationOfState.R * T / (Vm - b);
        double term2 = a / (Vm * (Vm + b) + b * (Vm - b));
        return term1 - term2;
    }

    @Override
    public List<Double> findSpinodalPoints(double T) {
        if (T >= Tc) return List.of();

        return findSpinodalPoints(T, b * 1.001f, 1e6);
    }
    @Override
    public List<Double> findSpinodalPoints(double T, double vMin, double vMax) {
        List<java.lang.Double> spinodals = new ArrayList<>();

        double previousV = vMin;
        double previousP = pressure(T, previousV);
        double previousSlope = java.lang.Double.NaN;

        for (double V = vMin; V <= vMax; V *= 1.05) {
            double P = pressure(T, V);
            double slope = (P - previousP) / (V - previousV);

            if (!java.lang.Double.isNaN(previousSlope) && previousSlope * slope <= 0) {
                // slope changed sign → inflection/spinodal
                spinodals.add(V);
            }

            previousV = V;
            previousP = P;
            previousSlope = slope;
            if (spinodals.size() == 2) break;
        }
        return spinodals;
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
        double A = a(T) * P / (EquationOfState.R * EquationOfState.R * T * T);
        double B = b * P / (EquationOfState.R * T);

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
        double R = EquationOfState.R;
        double a = a(T); // temperature-dependent 'a'
        double A = a * P / (R * R * T * T);
        double B = b * P / (R * T);
        double epsilon = 1e-9;
        if (B < epsilon) return 1.0; // Fugacity coefficient approaches 1 for ideal gas
        if (Math.abs(Z - B) < epsilon) return Double.NaN;

        double sqrt2 = Math.sqrt(2);

        double term1 = Z - 1 - Math.log(Z - B);
        double term2 = A / (2 * sqrt2 * B);
        double logArgument = (Z + (1 + sqrt2) * B) / (Z + (1 - sqrt2) * B);
        double term3 = term2 * Math.log(logArgument);

        double lnPhi = term1 - term3;

        return Math.exp(lnPhi);
    }
    public double saturationPressure(double T) {
        if (T >= Tc) {
            throw new IllegalArgumentException("T must be lower than the critical temperature to compute a coexistence pressure");
        }
        if (T <= 0) {
            throw new IllegalArgumentException("Temperature must be positive");
        }

        // Clamp below/above known range
        if (T <= saturationPressure.firstKey()) {
            return saturationPressure.get(saturationPressure.firstKey());
        }
        if (T >= saturationPressure.lastKey()) {
            return saturationPressure.get(saturationPressure.lastKey());
        }

        double index = T / TemperatureStep;
        int lowerIndex = (int) Math.floor(index);
        double frac = index - lowerIndex;

        float T1 = lowerIndex * TemperatureStep;
        float T2 = (lowerIndex + 1) * TemperatureStep;

        // Safeguard in case floating-point precision causes a missing key
        if (!saturationPressure.containsKey(T1) || !saturationPressure.containsKey(T2)) {
            Map.Entry<Float, Float> lower = ((TreeMap<Float, Float>) saturationPressure).floorEntry((float) T);
            Map.Entry<Float, Float> upper = ((TreeMap<Float, Float>) saturationPressure).ceilingEntry((float) T);

            if (lower == null || upper == null) {
                return saturationPressure.get(saturationPressure.firstKey());
            }

            float T_lower = lower.getKey();
            float T_upper = upper.getKey();
            if (T_lower == T_upper) {
                return saturationPressure.get(T_lower);
            }
            float fracAlt = (float)((T - T_lower) / (T_upper - T_lower));

            return lower.getValue() * (1 - fracAlt) + upper.getValue() * fracAlt;
        }

        double P1 = saturationPressure.get(T1);
        double P2 = saturationPressure.get(T2);

        return P1 * (1 - frac) + P2 * frac;
    }
    private double computeSaturationPressure(double T) {
        if (T >= Tc) {
            throw new IllegalArgumentException("T must be lower than the critical temperature to compute a coexistence pressure");
        }
        if (T <= 0) {
            throw new IllegalArgumentException("Temperature must be positive");
        }

        // Estimate pressure range based on spinodal points
        List<Double> spinodals = findSpinodalPoints(T);
        if (spinodals.size() < 2) {
            throw new RuntimeException("Not enough spinodal points found");
        }
        assert spinodals.size() == 2;

        double pMin = Math.max(1e-8, pressure(T, spinodals.get(0)));  // lower pressure bound
        double pMax = pressure(T, spinodals.get(1));               // upper pressure bound

        // Fugacity difference function
        Function<Float, Float> areaDifferenceFunction = (Float PGuess) -> {
            double[] roots = getZFactors(T, PGuess);
            if (roots.length < 2) return Float.MAX_VALUE;

            double Zl = Arrays.stream(roots).min().getAsDouble();
            double Zv = Arrays.stream(roots).max().getAsDouble();
            double Vl = Zl * EquationOfState.R * T / PGuess;
            double Vv = Zv * EquationOfState.R * T / PGuess;
            if (Math.abs(Vv - Vl) < 1e-9) return Float.MAX_VALUE;

            double area = 0;
            for (double V = Vl; V < Vv; V *= 1.01f) {
                double V2 = V*1.01f;
                double p1 = pressure(T, V);
                double p2 = pressure(T, V2);
                area += 0.5 * (p1 + p2) * (V2-V);
            }

            double rect = PGuess * (Vv - Vl);
            return (float)(area - rect);
        };
        try {
            return Math.max(1e-8,Solvers.dichotomy(areaDifferenceFunction, (float) pMin, (float) pMax, 200f));

        } catch (RuntimeException e) {
            return Math.max(1e-8,Solvers.gradientDecent((p )-> Math.abs(areaDifferenceFunction.apply(p)), (float) (pMin + pMax)/2, 100, 1, 2));

        }
    }


    public float saturationTemperature(float P) {
        if (P >= Pc) {
            throw new IllegalArgumentException("P must be lower than the critical pressure to compute a coexistence pressure");
        }
        if (P <= 0) {
            throw new IllegalArgumentException("Pressure must be positive");
        }

        double logP = Math.log(P);
        double scaledLogP = logP / PressureLogStep;
        int lowerIndex = (int) Math.floor(scaledLogP);
        double frac = scaledLogP - lowerIndex;

        float P1 = (float) Math.exp(lowerIndex * PressureLogStep);
        float P2 = (float) Math.exp((lowerIndex + 1) * PressureLogStep);

        if (P <= saturationTemperature.firstKey()) {
            return saturationTemperature.get(saturationTemperature.firstKey());
        }
        if (P >= saturationTemperature.lastKey()) {
            return saturationTemperature.get(saturationTemperature.lastKey());
        }

        // Safeguard against missing keys due to floating point precision
        if (!saturationTemperature.containsKey(P1) || !saturationTemperature.containsKey(P2)) {
            // Optionally, search for closest keys using ceiling/floor from TreeMap
            Map.Entry<Float, Float> lower = saturationTemperature.floorEntry(P);
            Map.Entry<Float, Float> upper = saturationTemperature.ceilingEntry(P);
            if (lower == null || upper == null) {
                return saturationTemperature.get(saturationTemperature.firstKey());
            }

            float logP1 = (float) Math.log(lower.getKey());
            float logP2 = (float) Math.log(upper.getKey());
            float fracAlt = (float)((logP - logP1) / (logP2 - logP1));
            return lower.getValue() * (1 - fracAlt) + upper.getValue() * fracAlt;
        }

        float T1 = saturationTemperature.get(P1);
        float T2 = saturationTemperature.get(P2);

        return (float) (T1 * (1 - frac) + T2 * frac);
    }

    protected  double residualEntropy(double T, double P, double Z) {
        /*double B = b * P / (R * T);
        double sqrt2 = Math.sqrt(2);
        double log1 = Math.log((Z - B) / Z);
        double log2 = Math.log((Z + (1 + sqrt2) * B) / (Z + (1 - sqrt2) * B));
        double da_dT = da_dT(T);

        double term1 = R * log1;
        double term2 = (1.0 / (2 * sqrt2 * b)) * da_dT * log2;

        return (term1 - term2)/M;*/
        double a = a(T);
        double sqrt2 = Math.sqrt(2);
        double B = b * P / (R * T);

        double term1 = R * Math.log(Z - B);
        double term2 = (a / (2 * sqrt2 * b * R * T)) *
                Math.log((Z + (1 + sqrt2) * B) / (Z + (1 - sqrt2) * B));

        return (term1 - term2)/M; // [J/mol·K]
    }

    @Override
    public double residualEnthalpy(double T, double P, double Z) {
        double a = a(T);
        double da_dT = da_dT(T); // must be implemented
        double B = b * P / (R * T);
        double sqrt2 = Math.sqrt(2);

        double logTerm = Math.log((Z + (1 + sqrt2) * B) / (Z + (1 - sqrt2) * B));
        double term1 = R * T * (Z - 1);
        double term2 = (T * da_dT - a) / (2 * sqrt2 * b) * logTerm;

        return (term1 + term2)/M; // [J/mol]
    }

    public double getB() {
        return b;
    }//this should be the starting point for plotting
    @Override
    public double minZ(float T, float P) {
        return (b *1.001f) * P / (EquationOfState.R * T); // Minimum physically valid Z
    }

    @Override
    public double volumeMolar(double T, double P, double vaporFraction) {
        try {
            return super.volumeMolar(T, P, vaporFraction);
        } catch (RuntimeException e) {
            return getB() * 1.001f;
        }
    }
}

