package com.rae.formicapi.new_thermalmodels;

import com.rae.formicapi.math.Solvers;
import com.rae.formicapi.math.data.ReversibleOneDTabulatedFunction;
import com.rae.formicapi.math.data.StepMode;
import com.rae.formicapi.thermal_utilities.eos.EquationOfState;
import org.jetbrains.annotations.NotNull;

import java.util.Arrays;
import java.util.List;

public class PengRobinsonEOS extends CubicEOS {

    private final double kappa, a0, b;
    private final ReversibleOneDTabulatedFunction saturationPressure;

    public PengRobinsonEOS(double Tc, double Pc, double omega, double M) {
        super(M, Tc, Pc);
        this.kappa = 0.37464 + 1.54226*omega - 0.26992*omega*omega;
        this.a0 = 0.45724 * R * R * Tc * Tc / Pc;
        this.b = 0.07780 * R * Tc / Pc;

        this.saturationPressure = new ReversibleOneDTabulatedFunction((T) -> (float) computeSaturationPressure(T), 100F,
                (float)Tc, StepMode.LINEAR, (float) (Tc/1e4f), StepMode.LOGARITHMIC, 0.01f);

    }

    private double alpha(double T) {
        double Tr = T / Tc;
        return Math.pow(1 + kappa*(1-Math.sqrt(Tr)), 2);
    }

    private double a(double T) { return a0 * alpha(T); }

    private double da_dT(double T) {
        double sqrtTr = Math.sqrt(T/Tc);
        return -a0 * kappa*(1+kappa*(1-sqrtTr))/(Tc*sqrtTr);
    }

    @Override
    public double pressure(double T, double Vm_mass) {
        double Vm = Vm_mass * M;
        double aT = a(T);
        double term1 = R*T/(Vm - b);
        double term2 = aT / (Vm*(Vm+b) + b*(Vm-b));
        return term1 - term2;
    }

    @Override
    public double dPdV(double T, double Vm_mass) {
        double Vm = Vm_mass * M;
        double aT = a(T);
        double denom = Math.pow(Vm*(Vm+b)+b*(Vm-b),2);
        return -R*T/Math.pow(Vm-b,2) + aT*(2*Vm + b)*(Vm - b)/denom;
    }

    @Override
    public double dPdT(double T, double Vm_mass) {
        double Vm = Vm_mass * M;
        double daT_dT = da_dT(T);
        return R/(Vm-b) - daT_dT / (Vm*(Vm+b) + b*(Vm-b));
    }

    @Override
    public double helmholtzFreeEnergy(double T, double Vm_mass) {
        double Vm = Vm_mass * M;
        double sqrt2 = Math.sqrt(2);
        double aT = a(T);
        double term1 = Math.log(Vm/(Vm-b));
        double term2 = Math.log((Vm + (1+sqrt2)*b)/(Vm + (1-sqrt2)*b));
        return (R*T*(term1 - aT/(2*sqrt2*b*R*T)*term2))/M;
    }

    @Override
    public double getM() { return M; }

    // ---------- ResidualProperties ----------
    @Override
    public double residualEntropy(double T, double Vm_mass) {
        double Vm_mol = Vm_mass * M;
        double sqrt2 = Math.sqrt(2);
        double daT_dT = da_dT(T);

        double logTerm = Math.log((Vm_mol + (1 + sqrt2) * b) / (Vm_mol + (1 - sqrt2) * b));
        double term1 = R * Math.log(Vm_mol / (Vm_mol - b));
        double term2 = daT_dT / (2 * sqrt2 * b) * logTerm;

        return (term1 - term2) / M;
    }

    @Override
    public double residualEnthalpy(double T, double Vm_mass) {
        double Vm_mol = Vm_mass * M;
        double sqrt2 = Math.sqrt(2);
        double aT = a(T);
        double daT_dT = da_dT(T);

        double logTerm = Math.log((Vm_mol + (1 + sqrt2) * b) / (Vm_mol + (1 - sqrt2) * b));
        double term1 = R * T * (Vm_mol / (Vm_mol - b) - 1);
        double term2 = (T * daT_dT - aT) / (2 * sqrt2 * b) * logTerm;

        return (term1 + term2) / M;
    }

    // ---------- Fugacity coefficient ----------
    @Override
    public double fugacityCoefficient(double T, double P, double Vm_mass) {
        double Vm = Vm_mass * M;
        double aT = a(T);
        double sqrt2 = Math.sqrt(2);
        double A = aT * P / (R * R * T * T);
        double B = b * P / (R * T);

        double term1 = Vm / (Vm - b) - Math.log(Vm / (Vm - b));
        double term2 = A / (2 * sqrt2 * B) * Math.log((Vm + (1 + sqrt2) * b) / (Vm + (1 - sqrt2) * b));

        return Math.exp(term1 - term2);
    }

    public double getB() {
        return b;
    }//this should be the starting point for plotting

    @Override
    public List<Double> findSpinodalPoints(double T) {
        if (T >= Tc) return List.of();

        return findSpinodalPoints(T, b * 1.001f / M, 1e6/M);
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
        double A = a(T) * P / (com.rae.formicapi.thermal_utilities.eos.EquationOfState.R * com.rae.formicapi.thermal_utilities.eos.EquationOfState.R * T * T);
        double B = b * P / (EquationOfState.R * T);

        // Coefficients of the cubic in Z
        double c1 = 1.0;
        double c2 = -(1.0 - B);
        double c3 = A - 3 * B * B - 2 * B;
        double c4 = -(A * B - B * B - B * B * B);

        // Solve cubic
        return Arrays.stream(Solvers.solveCubic(c1, c2, c3, c4))
                .filter(d -> !Double.isNaN(d) && d > 0)// && d > minZ((float) T, (float) P))
                .sorted()
                .toArray();
    }

    private double computeSaturationPressure(double T) {
        if (T >= Tc) {
            throw new IllegalArgumentException("T must be lower than the critical temperature to compute a coexistence pressure");
        }
        if (T <= 0) {
            throw new IllegalArgumentException("Temperature must be positive");
        }


        // 1) Get pressure bounds from spinodals
        List<Double> spinodals = findSpinodalPoints(T);
        if (spinodals.size() < 2) {
            throw new RuntimeException("Failed to find spinodal points");
        }

        double Pmin = Math.max(1e-6, pressure(T, spinodals.get(0)));
        double Pmax = pressure(T, spinodals.get(1));

        // 2) Fugacity equality function
        java.util.function.DoubleFunction<Double> f = (double P) -> {
            double[] Zs = getZFactors(T, P);
            if (Zs.length < 2) {
                return Double.NaN;
            }

            double Zl = Zs[0];
            double Zv = Zs[Zs.length - 1];

            double phil = fugacityCoefficient(T, P, Zl);
            double phiv = fugacityCoefficient(T, P, Zv);

            if (phil <= 0 || phiv <= 0 || Double.isNaN(phil) || Double.isNaN(phiv)) {
                return Double.NaN;
            }

            return Math.log(phil) - Math.log(phiv);
        };

        // 3) Bracket check
        double fmin = f.apply(Pmin);
        double fmax = f.apply(Pmax);

        if (Double.isNaN(fmin) || Double.isNaN(fmax) || fmin * fmax > 0) {
            throw new RuntimeException("Failed to bracket saturation pressure");
        }

        // 4) Bisection
        double P = 0.5 * (Pmin + Pmax);
        for (int i = 0; i < 80; i++) {
            double fp = f.apply(P);
            if (Math.abs(fp) < 1e-8) break;

            if (fp * fmin < 0) {
                Pmax = P;
                fmax = fp;
            } else {
                Pmin = P;
                fmin = fp;
            }

            P = 0.5 * (Pmin + Pmax);
        }

        return Math.max(P, 1e-6);
    }

    @Override
    public double saturationPressure(double T) {
        if (T >= Tc) {
            throw new IllegalArgumentException("T must be lower than the critical temperature to compute a coexistence pressure was "+ T + " Tc is "+ Tc);
        }
        if (T <= 0) {
            throw new IllegalArgumentException("Temperature must be strictly positive was " + T+ " did you sent a non kelvin temperature ?");
        }

        return saturationPressure.getF((float) T);
    }

    public float saturationTemperature(float P) {
        if (P >= Pc) {
            throw new IllegalArgumentException("P must be lower than the critical pressure to compute a coexistence pressure");
        }
        if (P <= 0) {
            throw new IllegalArgumentException("Pressure must be positive");
        }
        return saturationPressure.getInverseF(P);
    }
}