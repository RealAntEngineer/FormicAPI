package com.rae.formicapi.foundation.math.operators;

import java.util.Arrays;

/**
 * Fixed-structure sparse quadratic tensor.
 *
 * <p>Represents a quadratic nonlinear operator:
 *
 * <pre>
 *     F_i(x) = sum(c[i,j,k] * x[j] * x[k])
 * </pre>
 *
 * <p>This is the quadratic equivalent of {@link PaddedCSRMatrix}.
 *
 * <p>Storage is padded per equation:
 *
 * <pre>
 * equation 0:
 *     c0 * x[var1[0]] * x[var2[0]]
 *     c1 * x[var1[1]] * x[var2[1]]
 *
 * equation 1:
 *     ...
 * </pre>
 *
 * <p>The sparsity pattern is fixed after construction.
 */
public class PaddedCSR2Tensor {

    //this can get resized at will using a
    private       int equations;

    //those 2 are supposed to be constant
    private final int variables;
    private final int termsPerEquation;

    /**
     * Quadratic coefficients.
     */
    private double[] values;

    /**
     * First variable index of each quadratic term.
     */
    private int[] var1Index;

    /**
     * Second variable index of each quadratic term.
     */
    private int[] var2Index;


    /**
     * Creates an empty quadratic tensor.
     *
     * @param equations number of output equations
     * @param variables number of unknown variables
     * @param termsPerEquation fixed number of quadratic terms per equation
     */
    public PaddedCSR2Tensor(int equations, int variables, int termsPerEquation) {
        this.equations = equations;
        this.variables = variables;
        this.termsPerEquation = termsPerEquation;

        int size = equations * termsPerEquation;

        this.values = new double[size];
        this.var1Index = new int[size];
        this.var2Index = new int[size];
    }


    /**
     * Replaces one equation's quadratic structure.
     *
     * @param equation equation index
     * @param newValues coefficients
     * @param newVar1 first variable indices
     * @param newVar2 second variable indices
     * @param count number of active terms
     */
    public void setRow(int equation, double[] newValues, int[] newVar1,int[] newVar2, int count) {

        if (count > termsPerEquation)
            throw new IllegalArgumentException(
                    "Too many terms: " + count
            );

        if (newVar1.length < count ||
                newVar2.length < count ||
                newValues.length < count)
            throw new IllegalArgumentException(
                    "Input arrays shorter than count"
            );


        int base = equation * termsPerEquation;

        for (int i = 0; i < count; i++) {
            values[base + i] = newValues[i];
            var1Index[base + i] = newVar1[i];
            var2Index[base + i] = newVar2[i];
        }


        // Disable unused terms
        for (int i = count; i < termsPerEquation; i++) {
            values[base + i] = 0.0;

            // safe default indices
            var1Index[base + i] = 0;
            var2Index[base + i] = 0;
        }
    }


    /**
     * Adds a coefficient to an existing quadratic term.
     *
     * @throws IllegalStateException if the term does not exist
     */
    public void add(int equation, int var1, int var2, double value) {

        if (value == 0.0)
            return;


        int base = equation * termsPerEquation;

        for (int i = 0; i < termsPerEquation; i++) {

            int idx = base + i;

            if (var1Index[idx] == var1 &&
                    var2Index[idx] == var2) {

                values[idx] += value;
                return;
            }
        }


        throw new IllegalStateException(
                "Tensor entry does not exist: (" +
                        equation + "," +
                        var1 + "," +
                        var2 + ")"
        );
    }


    /**
     * Evaluates:
     *
     * <pre>
     * result = F(x)
     *
     * F_i = sum(c*x_j*x_k)
     * </pre>
     */
    public void multiply(double[] x, double[] result) {

        for (int row = 0; row < equations; row++) {

            double sum = 0.0;

            int base = row * termsPerEquation;

            for (int i = 0; i < termsPerEquation; i++) {

                int idx = base + i;

                sum += values[idx]
                        * x[var1Index[idx]]
                        * x[var2Index[idx]];
            }

            result[row] = sum;
        }
    }

    /**
     * Evaluates the Jacobian-vector product:
     *
     * <pre>
     * result = J(x) * direction
     *
     * For:
     *     F_i = sum(c*x_j*x_k)
     *
     * the contribution of one term is:
     *
     *     c * (direction_j * x_k + x_j * direction_k)
     * </pre>
     *
     * <p>This avoids constructing the Jacobian matrix explicitly.
     *
     * @param x point where the Jacobian is evaluated
     * @param direction vector multiplied by the Jacobian
     * @param result output vector; length must be at least {@code equations}
     */
    public void multiplyJacobian(double[] x, double[] direction, double[] result) {

        for (int row = 0; row < equations; row++) {

            double sum = 0.0;

            int base = row * termsPerEquation;

            for (int i = 0; i < termsPerEquation; i++) {

                int idx = base + i;

                double c = values[idx];

                int j = var1Index[idx];
                int k = var2Index[idx];

                if (j == k) {
                    // d(c*x_j*x_j) = 2*c*x_j*dx_j
                    sum += 2.0 * c * x[j] * direction[j];
                } else {
                    // d(c*x_j*x_k) = c*(x_k*dx_j + x_j*dx_k)
                    sum += c * (
                            direction[j] * x[k]
                                    + x[j] * direction[k]
                    );
                }
            }

            result[row] = sum;
        }
    }

    public int equations() {
        return equations;
    }


    public int variables() {
        return variables;
    }


    public int termsPerEquation() {
        return termsPerEquation;
    }


    public void resize(int newEquations) {

        int required = newEquations * termsPerEquation;

        if (required > values.length) {

            values = Arrays.copyOf(values, required);
            var1Index = Arrays.copyOf(var1Index, required);
            var2Index = Arrays.copyOf(var2Index, required);
        }

        equations = newEquations;
    }
}
