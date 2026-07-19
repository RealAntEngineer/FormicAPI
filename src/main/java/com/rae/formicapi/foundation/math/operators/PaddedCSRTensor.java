package com.rae.formicapi.foundation.math.operators;

import java.util.Arrays;

/**
 * Fixed-structure sparse N-order tensor.
 *
 * <p>Represents:
 *
 * <pre>
 * F_i(x) = sum(c * x[j1] * x[j2] * ... * x[jN])
 * </pre>
 *
 * <p>Storage is padded per equation.
 *
 * <p>The sparsity pattern is fixed after construction.
 */
public class PaddedCSRTensor {

    private final int termsPerEquation;
    private final int order;
    private       int equations;
    /**
     * Tensor coefficients.
     */
    private double[] values;

    /**
     * Variable indices.
     *
     * <p>First index is the term index.
     * Second index is the tensor dimension.
     *
     * <pre>
     * variableIndices[term][dimension]
     * </pre>
     */
    private int[][] variableIndices;


    public PaddedCSRTensor(int order, int equations, int termsPerEquation) {

        if (order < 1)
            throw new IllegalArgumentException("Order must be positive");

        this.equations = equations;
        this.termsPerEquation = termsPerEquation;
        this.order = order;

        int size = equations * termsPerEquation;

        this.values = new double[size];

        this.variableIndices = new int[size][order];
    }

    /**
     * Overwrites an entire equation of the tensor, including both its structure
     * (variable indices of each tensor term) and its coefficients.
     *
     * <p>This method assumes a fixed storage capacity defined by
     * {@code termsPerEquation}. The provided {@code varIndices} array defines
     * the variable indices for each tensor term, and {@code newValues} defines
     * the corresponding coefficients.
     *
     * <p>After this call:
     * <ul>
     *     <li>The previous contents of the equation are fully replaced</li>
     *     <li>The tensor structure of the equation is updated to match
     *         {@code varIndices}</li>
     *     <li>No resizing of the underlying storage occurs</li>
     * </ul>
     *
     * <p><b>Important:</b> This operation modifies the sparsity structure of the
     * tensor equation. All subsequent operations (e.g., tensor evaluation and
     * Jacobian-vector products) will use the updated term layout.
     *
     * <p>Each tensor term is defined by one coefficient and {@code order}
     * variable indices:
     *
     * <pre>
     * varIndices[term][dimension]
     *
     * coefficient * x[varIndices[term][0]]
     *             * x[varIndices[term][1]]
     *             * ...
     *             * x[varIndices[term][order-1]]
     * </pre>
     *
     * <p>Constraints:
     * <ul>
     *     <li>{@code newValues.length >= count}</li>
     *     <li>{@code varIndices.length == count}</li>
     *     <li>{@code varIndices[i].length == order} for every term</li>
     *     <li>{@code count <= termsPerEquation}</li>
     * </ul>
     *
     * <p>Unused terms after {@code count} are disabled and their coefficients
     * are set to zero.
     *
     * <p>Performance: O(count * order)
     *
     * @param equation   equation index to modify
     * @param newValues  coefficients of the tensor terms
     * @param varIndices variable indices of each tensor term,
     *                   with shape {@code [count][order]}
     * @param count      number of active tensor terms in this equation
     * @throws IllegalArgumentException if array sizes do not match the tensor
     *                                  order or if {@code count} exceeds
     *                                  {@code termsPerEquation}
     */
    public void setRow(int equation, double[] newValues, int[][] varIndices, int count) {

        if (count > termsPerEquation) throw new IllegalArgumentException("Too many terms: " + count);

        if (newValues.length < count) throw new IllegalArgumentException("Values shorter than count");

        for (int i = 0; i < count; i++) {
            if (varIndices[i].length != order)
                throw new IllegalArgumentException("Wrong tensor order");
        }

        int base = equation * termsPerEquation;

        for (int i = 0; i < count; i++) {

            if (varIndices[i].length != order)
                throw new IllegalArgumentException("Wrong tensor order");

            values[base + i] = newValues[i];

            System.arraycopy(varIndices[i], 0, variableIndices[base + i], 0, order);
        }

        // disable unused entries
        for (int i = count; i < termsPerEquation; i++) {
            values[base + i] = 0.0;
            Arrays.fill(variableIndices[base + i], 0);
        }
    }


    /**
     * Adds a coefficient to an existing tensor term.
     */
    public void add(int equation, double value, int... vars) {

        if (value == 0.0)
            return;

        if (vars.length != order)
            throw new IllegalArgumentException("Wrong tensor order");

        int base = equation * termsPerEquation;

        for (int i = 0; i < termsPerEquation; i++) {

            int idx = base + i;

            if (Arrays.equals(variableIndices[idx], vars)) {
                values[idx] += value;
                return;
            }
        }

        throw new IllegalStateException("Tensor entry does not exist");
    }

    /**
     * Evaluates:
     *
     * <pre>
     * result = F(x)
     * </pre>
     */
    public void multiply(double[] x, double[] result) {

        for (int row = 0; row < equations; row++) {

            double sum  = 0.0;
            int    base = row * termsPerEquation;

            for (int i = 0; i < termsPerEquation; i++) {

                int    idx = base + i;
                double p   = values[idx];

                for (int d = 0; d < order; d++)
                    p *= x[variableIndices[idx][d]];

                sum += p;
            }

            result[row] = sum;
        }
    }

    /**
     * Evaluates:
     *
     * <pre>
     * result = J(x) * direction
     * </pre>
     */
    public void multiplyJacobian(double[] x, double[] direction, double[] result) {

        for (int row = 0; row < equations; row++) {

            double sum  = 0.0;
            int    base = row * termsPerEquation;

            for (int i = 0; i < termsPerEquation; i++) {

                int idx = base + i;

                for (int m = 0; m < order; m++) {

                    double term = values[idx];

                    for (int r = 0; r < order; r++) {

                        int v = variableIndices[idx][r];

                        if (r == m)
                            term *= direction[v];
                        else
                            term *= x[v];
                    }
                    sum += term;
                }
            }
            result[row] = sum;
        }
    }


    public int equations() {
        return equations;
    }

    public int termsPerEquation() {
        return termsPerEquation;
    }

    public int order() {
        return order;
    }

    public void resize(int newEquations) {

        int required = newEquations * termsPerEquation;

        if (required > values.length) {

            values = Arrays.copyOf(values, required);

            int[][] newIndices = new int[required][order];

            for (int i = 0; i < variableIndices.length; i++)
                System.arraycopy(variableIndices[i], 0, newIndices[i], 0, order);

            variableIndices = newIndices;
        }
        equations = newEquations;
    }
}