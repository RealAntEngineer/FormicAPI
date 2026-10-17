package com.rae.formicapi.content.thermal_utilities;

import java.util.EnumMap;

public class PolynomeBased {

    float Tc; // unused: "T >= Tc" is replaced by "c >= c(T = Tc, P)", T being monotonic in h and s
    float Pc;

    // Curves of P (JSON "curves"). The variable is computed here:
    // SatL / SatV use uBelow, the Tc curves use uTc (one curve on both sides of Pc).
    Polynomial1D PHSatV, PHSatL, PHTc; // hSatV, hSatL, hTc (h on the T = Tc isotherm)
    Polynomial1D PSSatV, PSSatL, PSTc; // same in s

    // One 2D polynomial per single phase (JSON "models"). The dome has none: it is a mix.
    EnumMap<Phase, Polynomial2D> PHTFunc;   // T   (h, lnP)
    EnumMap<Phase, Polynomial2D> PHSFunc;   // S   (h, lnP)
    EnumMap<Phase, Polynomial2D> PHrhoFunc; // rho (h, lnP)
    EnumMap<Phase, Polynomial2D> PSHFunc;   // H   (s, lnP)

    private double uBelow(float P) {
        return Math.sqrt(Math.max(Math.log(Pc / (double) P), 0.0));
    }

    /** JSON variable "signed_fifth_root": sign(d) * |d|^(1/5), d = ln(P/Pc). Negative below Pc. */
    private double uTc(float P) {
        double d = Math.log(P / (double) Pc);
        return Math.signum(d) * Math.pow(Math.abs(d), 0.2);
    }

    /** h / s of a dome component (LIQUID or VAPOR) at saturation. */
    private double hSat(Phase phase, float P) {
        return (phase == Phase.LIQUID ? PHSatL : PHSatV).evaluate(uBelow(P));
    }

    private double sSat(Phase phase, float P) {
        return (phase == Phase.LIQUID ? PSSatL : PSSatV).evaluate(uBelow(P));
    }

    // In a mix (dome) every component is evaluated at its own saturation point;
    // a single phase is evaluated at the state itself.

    public float getT(float P, float H) {
        Phase.PhaseMix mix = phaseH(P, H);
        double lnP = Math.log(P);
        double T = 0;
        for (int i = 0; i < mix.phases().length; i++) {
            Phase phase = mix.phases()[i];
            double h = mix.phases().length == 1 ? H : hSat(phase, P);//we need to get the saturated lines
            T += mix.fractions()[i] * PHTFunc.get(phase).evaluate(h, lnP);
        }
        return (float) T;
    }

    public float getS(float P, float H) {
        Phase.PhaseMix mix = phaseH(P, H);
        double lnP = Math.log(P);
        double S = 0;
        for (int i = 0; i < mix.phases().length; i++) {
            Phase phase = mix.phases()[i];
            double h = mix.phases().length == 1 ? H : hSat(phase, P);
            S += mix.fractions()[i] * PHSFunc.get(phase).evaluate(h, lnP);
        }
        return (float) S;
    }

    public float getH(float P, float S) {
        Phase.PhaseMix mix = phaseS(P, S);
        double lnP = Math.log(P);
        double H = 0;
        for (int i = 0; i < mix.phases().length; i++) {
            Phase phase = mix.phases()[i];
            double s = mix.phases().length == 1 ? S : sSat(phase, P);
            H += mix.fractions()[i] * PSHFunc.get(phase).evaluate(s, lnP);
        }
        return (float) H;
    }

    public float getRho(float P, float H) {
        Phase.PhaseMix mix = phaseH(P, H);
        double lnP = Math.log(P);
        double v = 0; // specific volume mixes linearly: 1/rho = sum x_i / rho_i
        for (int i = 0; i < mix.phases().length; i++) {
            Phase phase = mix.phases()[i];
            double h = mix.phases().length == 1 ? H : hSat(phase, P);
            v += mix.fractions()[i] / PHrhoFunc.get(phase).evaluate(h, lnP);
        }
        return (float) (1.0 / v);
    }

    public Phase.PhaseMix phaseH(float P, float H) {

        //for now no solid (too complex)
        //"T >= Tc" <=> "H >= h(T = Tc, P)"
        float hTc = (float) PHTc.evaluate(uTc(P));

        if (P >= Pc && H >= hTc) {
            return new Phase.PhaseMix(Phase.SUPERCRITICAL);
        } else if (P >= Pc && H < hTc) {
            return new Phase.PhaseMix(Phase.SUPERCRITICAL_LIQUID);
        } else if (P < Pc && H >= hTc) {
            return new Phase.PhaseMix(Phase.SUPERCRITICAL_VAPOR);
        } else {
            float hV = (float) PHSatV.evaluate(uBelow(P));
            float hL = (float) PHSatL.evaluate(uBelow(P));
            //safeguard against 0
            if (Math.abs(hL - hV) < 1e-8) return new Phase.PhaseMix(Phase.SUPERCRITICAL);

            if (H > hV) {
                return new Phase.PhaseMix(Phase.VAPOR);
            } else if (H < hL) {
                return new Phase.PhaseMix(Phase.LIQUID);
            }
            return new Phase.PhaseMix(
                    new Phase[]{Phase.LIQUID, Phase.VAPOR},
                    new float[]{1 - (H - hL) / (hV - hL), (H - hL) / (hV - hL)
                    });
        }
    }

    public Phase.PhaseMix phaseS(float P, float S) {

        //for now no solid (too complex)
        //"T >= Tc" <=> "S >= s(T = Tc, P)"
        float sTc = (float) PSTc.evaluate(uTc(P));

        if (P >= Pc && S >= sTc) {
            return new Phase.PhaseMix(Phase.SUPERCRITICAL);
        } else if (P >= Pc && S < sTc) {
            return new Phase.PhaseMix(Phase.SUPERCRITICAL_LIQUID);
        } else if (P < Pc && S >= sTc) {
            return new Phase.PhaseMix(Phase.SUPERCRITICAL_VAPOR);
        } else {
            float sV = (float) PSSatV.evaluate(uBelow(P));
            float sL = (float) PSSatL.evaluate(uBelow(P));
            //safeguard against 0
            if (Math.abs(sL - sV) < 1e-8) return new Phase.PhaseMix(Phase.SUPERCRITICAL);

            if (S > sV) {
                return new Phase.PhaseMix(Phase.VAPOR);
            } else if (S < sL) {
                return new Phase.PhaseMix(Phase.LIQUID);
            }
            return new Phase.PhaseMix(
                    new Phase[]{Phase.LIQUID, Phase.VAPOR},
                    new float[]{1 - (S - sL) / (sV - sL), (S - sL) / (sV - sL)
                    });
        }
    }
}