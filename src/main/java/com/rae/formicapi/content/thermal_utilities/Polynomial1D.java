package com.rae.formicapi.content.thermal_utilities;

/**
 * 1D polynomial
 *   u  = clamp(u, xMin, xMax)                   (same role as a clamped table lookup)
 *   un = (u - xCenter) / xScale
 *   f  = sum_k c[k] * un^k                      (Horner)
 *   f  = exp(f)  if logOutput
 *
 * All maths is done in double: with order ~12 the coefficients can be large and
 * alternate in sign, which float32 would not survive. Only the public API is float.
 */
public class Polynomial1D {

    private final double[] coefficients; // coefficients[k] multiplies un^k
    private final double xCenter;
    private final double xScale;
    private final double xMin;
    private final double xMax;

    /**
     * @param coefficients JSON "coefficients" (index = power)
     * @param xCenter      JSON "x_center"
     * @param xScale       JSON "x_scale"
     * @param xMin         JSON "x_min"
     * @param xMax         JSON "x_max"
     */
    public Polynomial1D(double[] coefficients, double xCenter, double xScale, double xMin, double xMax) {
        if (coefficients == null || coefficients.length == 0) {
            throw new IllegalArgumentException("Polynomial1D needs at least one coefficient");
        }
        if (xScale == 0) {
            throw new IllegalArgumentException("xScale must be non-zero");
        }
        this.coefficients = coefficients.clone();
        this.xCenter = xCenter;
        this.xScale = xScale;
        this.xMin = xMin;
        this.xMax = xMax;
    }

    //TODO generalise by moving that outside of evaluate
    /** Evaluate at ln(P), P in Pa. */
    public double evaluate(double u) {
        u = Math.clamp(u, xMin, xMax);
        double un = (u - xCenter) / xScale;

        double r = 0.0;
        for (int k = coefficients.length - 1; k >= 0; k--) {
            r = r * un + coefficients[k];
        }
        return r;
    }

    public int getOrder() {
        return coefficients.length - 1;
    }
}