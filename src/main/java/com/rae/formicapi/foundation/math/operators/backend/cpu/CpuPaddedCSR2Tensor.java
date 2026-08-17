package com.rae.formicapi.foundation.math.operators.backend.cpu;

import com.rae.formicapi.foundation.math.operators.vectors.Vector;

import java.util.Arrays;

/**
 * CPU backend for {@link com.rae.formicapi.foundation.math.operators.nonlinear.PaddedCSR2Tensor}.
 *
 * <p>Fixed-structure sparse quadratic tensor:
 *
 * <pre>
 * F_i(x) = sum(c[i,j,k] * x[j] * x[k])
 * </pre>
 *
 * <p>This mirrors {@link CpuPaddedCSRMatrix}: storage/structure mutation
 * ({@link #setRow}, {@link #add}, {@link #resize}) is identical to the plain
 * tensor, but evaluation goes through {@link Vector}s and is parallelized
 * across equations via a {@link CpuExecutor} once the row count crosses the
 * executor's parallel threshold.
 *
 * <p>Storage layout is equation-major (same GPU caveat as
 * {@code PaddedCSR2Tensor}: a term-major layout would be preferable for a
 * GPU backend, but is fine here since threads/tasks are split by row range).
 */
public class CpuPaddedCSR2Tensor {

    CpuExecutor executor;

    private int equations;
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
     * @param equations        number of output equations
     * @param termsPerEquation fixed number of quadratic terms per equation
     */
    public CpuPaddedCSR2Tensor(int equations, int termsPerEquation) {
        this.equations = equations;
        this.termsPerEquation = termsPerEquation;

        int size = equations * termsPerEquation;

        this.values = new double[size];
        this.var1Index = new int[size];
        this.var2Index = new int[size];
    }

    /**
     * Replaces one equation's quadratic structure.
     *
     * <p>See {@code PaddedCSR2Tensor#setRow} for the full contract; behaves
     * identically here.
     *
     * @param equation  equation index
     * @param newValues coefficients
     * @param var1Index first variable indices
     * @param var2Index second variable indices
     * @param count     number of active terms
     */
    public void setRow(int equation, double[] newValues, int[] var1Index, int[] var2Index, int count) {

        if (count > termsPerEquation)
            throw new IllegalArgumentException("Too many terms: " + count);

        if (var1Index.length < count ||
                var2Index.length < count ||
                newValues.length < count)
            throw new IllegalArgumentException("Input arrays shorter than count");

        int base = equation * termsPerEquation;

        for (int i = 0; i < count; i++) {
            values[base + i] = newValues[i];
            this.var1Index[base + i] = var1Index[i];
            this.var2Index[base + i] = var2Index[i];
        }

        // Disable unused terms
        for (int i = count; i < termsPerEquation; i++) {
            values[base + i] = 0.0;
            this.var1Index[base + i] = 0;
            this.var2Index[base + i] = 0;
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

            if (var1Index[idx] == var1 && var2Index[idx] == var2) {
                values[idx] += value;
                return;
            }
        }

        throw new IllegalStateException(
                "Tensor entry does not exist: (" + equation + "," + var1 + "," + var2 + ")"
        );
    }

    /**
     * Evaluates {@code result = F(x)}, dispatching serially or in parallel
     * across equations depending on {@link CpuExecutor#getParallelThreshold()}.
     */
    public void apply(Vector x, Vector result) {
        if (executor == null) throw new RuntimeException("Executor wasn't setup");
        if (x instanceof CpuDoubleVector xCpu && result instanceof CpuDoubleVector resCpu) {
            double[] xArr = xCpu.array();
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

        final double[] values = this.values;
        final int[] var1Index = this.var1Index;
        final int[] var2Index = this.var2Index;

        for (int row = start; row < end; row++) {

            double sum = 0.0;

            int base = row * termsPerEquation;
            int rowEnd = base + termsPerEquation;

            for (int idx = base; idx < rowEnd; idx++) {
                sum += values[idx] * xArr[var1Index[idx]] * xArr[var2Index[idx]];
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
     *
     * <p>For {@code F_i = sum(c*x_j*x_k)} the contribution of one term is
     * {@code c * (direction_j * x_k + x_j * direction_k)}; this also handles
     * {@code j == k} correctly without a special case.
     */
    public void applyJacobian(Vector x, Vector direction, Vector result) {
        if (executor == null) throw new RuntimeException("Executor wasn't setup");
        if (x instanceof CpuDoubleVector xCpu &&
                direction instanceof CpuDoubleVector dCpu &&
                result instanceof CpuDoubleVector resCpu) {

            double[] xArr = xCpu.array();
            double[] dArr = dCpu.array();
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

        final double[] values = this.values;
        final int[] var1Index = this.var1Index;
        final int[] var2Index = this.var2Index;

        for (int row = start; row < end; row++) {

            double sum = 0.0;

            int idx = row * termsPerEquation;
            int rowEnd = idx + termsPerEquation;

            while (idx < rowEnd) {

                double c = values[idx];
                int j = var1Index[idx];
                int k = var2Index[idx];

                sum += c * (dArr[j] * xArr[k] + xArr[j] * dArr[k]);
                idx++;
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

    public void resize(int newEquations) {

        int required = newEquations * termsPerEquation;

        if (required > values.length || required > var1Index.length || required > var2Index.length) {

            values = Arrays.copyOf(values, required);
            var1Index = Arrays.copyOf(var1Index, required);
            var2Index = Arrays.copyOf(var2Index, required);
        }

        equations = newEquations;
    }

    public double[] getRowValues(int row) {
        int base = row * termsPerEquation;

        double[] out = new double[termsPerEquation];
        System.arraycopy(values, base, out, 0, termsPerEquation);

        return out;
    }

    public int[] getRowVar1(int row) {
        int base = row * termsPerEquation;

        int[] out = new int[termsPerEquation];
        System.arraycopy(var1Index, base, out, 0, termsPerEquation);

        return out;
    }

    public int[] getRowVar2(int row) {
        int base = row * termsPerEquation;

        int[] out = new int[termsPerEquation];
        System.arraycopy(var2Index, base, out, 0, termsPerEquation);

        return out;
    }

    public void setExecutor(CpuExecutor cpuExecutor) {
        executor = cpuExecutor;
    }
}