package com.rae.formicapi.foundation.math.operators.backend.cpu;

import com.rae.formicapi.foundation.math.operators.vectors.Vector;

import java.util.Arrays;

/**
 * CPU backend for {@link com.rae.formicapi.foundation.math.operators.nonlinear.PaddedCSRTensor}.
 *
 * <p>Fixed-structure sparse N-order tensor:
 *
 * <pre>
 * F_i(x) = sum(c * x[j1] * x[j2] * ... * x[jN])
 * </pre>
 *
 * <p>This mirrors {@link CpuPaddedCSRMatrix} and {@link CpuPaddedCSR2Tensor}:
 * storage/structure mutation ({@link #setRow}, {@link #add}, {@link #resize})
 * is identical to the plain tensor, but evaluation goes through
 * {@link Vector}s and is parallelized across equations via a
 * {@link CpuExecutor} once the row count crosses the executor's parallel
 * threshold.
 */
public class CpuPaddedCSRTensor extends CpuExecutable {

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
     * <p>{@code variableIndices[term][dimension]}
     */
    private int[][] variableIndices;

    public CpuPaddedCSRTensor(int order, int equations, int termsPerEquation) {

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
     * Overwrites an entire equation of the tensor, including both its
     * structure (variable indices of each tensor term) and its coefficients.
     *
     * <p>See {@code PaddedCSRTensor#setRow} for the full contract; behaves
     * identically here.
     *
     * @param equation   equation index to modify
     * @param newValues  coefficients of the tensor terms
     * @param varIndices variable indices of each tensor term, shape {@code [count][order]}
     * @param count      number of active tensor terms in this equation
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
     * Evaluates {@code result = F(x)}, dispatching serially or in parallel
     * across equations depending on {@link CpuExecutor#getParallelThreshold()}.
     */
    public void apply(Vector x, Vector result) {
        if (executor == null) throw new RuntimeException("Executor wasn't setup");
        if (x instanceof CpuDoubleVector xCpu && result instanceof CpuDoubleVector resCpu) {
            double[] xArr      = xCpu.array();
            double[] resultArr = resCpu.array();

            if (equations < executor.getParallelThreshold()) {
                applyRange(0, equations, xArr, resultArr);
                return;
            }

            executor.parallelFor(equations, (start, end) -> applyRange(start, end, xArr, resultArr));
        } else {
            throw new IllegalArgumentException("For a cpu backend tensor, you need to cpu backend vectors");
        }
    }

    private void applyRange(int start, int end, double[] xArr, double[] resultArr) {

        final double[] values          = this.values;
        final int[][]  variableIndices = this.variableIndices;
        final int      order           = this.order;

        for (int row = start; row < end; row++) {

            double sum  = 0.0;
            int    base = row * termsPerEquation;

            for (int i = 0; i < termsPerEquation; i++) {

                int    idx = base + i;
                double p   = values[idx];

                for (int d = 0; d < order; d++)
                    p *= xArr[variableIndices[idx][d]];

                sum += p;
            }

            resultArr[row] = sum;
        }
    }

    public void multiply(double[] x, double[] result) {
        throw new RuntimeException("Unsuported, use vector version instead");
    }

    /**
     * Evaluates the Jacobian-vector product {@code result = J(x) * direction},
     * dispatching serially or in parallel across equations depending on
     * {@link CpuExecutor#getParallelThreshold()}.
     */
    public void applyJacobian(Vector x, Vector direction, Vector result) {
        if (executor == null) throw new RuntimeException("Executor wasn't setup");
        if (x instanceof CpuDoubleVector xCpu &&
                direction instanceof CpuDoubleVector dCpu &&
                result instanceof CpuDoubleVector resCpu) {

            double[] xArr      = xCpu.array();
            double[] dArr      = dCpu.array();
            double[] resultArr = resCpu.array();

            if (equations < executor.getParallelThreshold()) {
                applyJacobianRange(0, equations, xArr, dArr, resultArr);
                return;
            }

            executor.parallelFor(equations, (start, end) -> applyJacobianRange(start, end, xArr, dArr, resultArr));
        } else {
            throw new IllegalArgumentException("For a cpu backend tensor, you need to cpu backend vectors");
        }
    }

    private void applyJacobianRange(int start, int end, double[] xArr, double[] dArr, double[] resultArr) {

        final double[] values          = this.values;
        final int[][]  variableIndices = this.variableIndices;
        final int      order           = this.order;

        for (int row = start; row < end; row++) {

            double sum  = 0.0;
            int    base = row * termsPerEquation;

            for (int i = 0; i < termsPerEquation; i++) {

                int idx = base + i;

                for (int m = 0; m < order; m++) {

                    double term = values[idx];

                    for (int r = 0; r < order; r++) {

                        int v = variableIndices[idx][r];

                        if (r == m)
                            term *= dArr[v];
                        else
                            term *= xArr[v];
                    }
                    sum += term;
                }
            }
            resultArr[row] = sum;
        }
    }

    public void multiplyJacobian(double[] x, double[] direction, double[] result) {
        throw new RuntimeException("Unsuported, use vector version instead");
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