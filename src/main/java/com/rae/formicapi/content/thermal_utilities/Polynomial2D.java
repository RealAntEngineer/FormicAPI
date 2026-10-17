package com.rae.formicapi.content.thermal_utilities;

/**
 * 2D polynomial in (x, y) with total degree &lt;= order, as exported by
 * train_poly_eos_v2.py ("models" -> property -> "phases" -> phase).
 * In the EOS, x is h (or s) and y is ln(P).
 *
 * <pre>
 *    xn = (x - xCenter) / xScale
 *    yn = (y - yCenter) / yScale
 *    f  = sum c[k] * xn^i * yn^j       with i + j &lt;= order
 *    f  = exp(f)  if logOutput
 * </pre>
 * <p>
 * Coefficient layout (flat array, same as the JSON): i is the OUTER loop
 * (power of x, 0..order), j the INNER loop (power of y, 0..order-i):
 * <pre>
 *   k = 0; for i in 0..order: for j in 0..order-i: c[k++] multiplies xn^i * yn^j
 * </pre>
 * Length = (order+1)(order+2)/2.
 * <p>
 * All maths is done in double (order-5 terms in float32 lose precision); only the
 * convenience API is float.
 */
public class Polynomial2D {


    private final int      order;
    private final double[] coefficients;
    private final int[]    rowStart; // rowStart[i] = index of the (i, j=0) coefficient
    private final double   xCenter;
    private final double   xScale;
    private final double   yCenter;
    private final double yScale;

    /**
     * @param order        JSON "order"
     * @param coefficients JSON "coefficients" (flat, layout above)
     * @param xCenter      JSON "x_center"
     * @param xScale       JSON "x_scale"
     * @param yCenter      JSON "y_center"
     * @param yScale       JSON "y_scale"
     */
    public Polynomial2D(int order, double[] coefficients, double xCenter, double xScale, double yCenter, double yScale) {
        int expected = (order + 1) * (order + 2) / 2;
        if (coefficients == null || coefficients.length != expected) {
            throw new IllegalArgumentException("Polynomial2D of order " + order + " needs "
                    + expected + " coefficients, got " + (coefficients == null ? 0 : coefficients.length));
        }
        if (xScale == 0 || yScale == 0) {
            throw new IllegalArgumentException("scales must be non-zero");
        }
        this.order = order;
        this.coefficients = coefficients.clone();
        this.xCenter = xCenter;
        this.xScale = xScale;
        this.yCenter = yCenter;
        this.yScale = yScale;

        this.rowStart = new int[order + 1];
        int k = 0;
        for (int i = 0; i <= order; i++) {
            rowStart[i] = k;
            k += order - i + 1;
        }
    }

    public double[] getCoefficients() {
        return coefficients;
    }

    public double getxCenter() {
        return xCenter;
    }

    public double getxScale() {
        return xScale;
    }

    public double getyCenter() {
        return yCenter;
    }

    public double getyScale() {
        return yScale;
    }

    /**
     * Evaluate with y already given as ln(P).
     */
    public double evaluate(double x, double y) {
        double xn = (x - xCenter) / xScale;
        double yn = (y - yCenter) / yScale;

        // Outer Horner over i (power of x), inner Horner over j (power of y). No allocation.
        double outer = 0.0;
        for (int i = order; i >= 0; i--) {
            int    start = rowStart[i];
            double inner = 0.0;
            for (int j = order - i; j >= 0; j--) {
                inner = inner * yn + coefficients[start + j];
            }
            outer = outer * xn + inner;
        }
        return outer;
    }

    public int getOrder() {
        return order;
    }
}