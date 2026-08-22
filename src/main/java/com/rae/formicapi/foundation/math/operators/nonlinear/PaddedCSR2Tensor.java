package com.rae.formicapi.foundation.math.operators.nonlinear;

import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuDoubleVector;
import com.rae.formicapi.foundation.math.operators.linear.PaddedCSRMatrix;
import com.rae.formicapi.foundation.math.operators.vectors.DoubleVector;

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
public class PaddedCSR2Tensor implements NonlinearOperator {

    private       int equations;

    private final int termsPerEquation;

    /**
     * Quadratic coefficients.
     */
    private double[] values;


    //pack both indexes into a long for faster access ?
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
     * @param termsPerEquation fixed number of quadratic terms per equation
     */
    public PaddedCSR2Tensor(int equations, int termsPerEquation) {
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
     * <p><b>GPU note:</b> The current storage layout is equation-major:
     * all terms belonging to one equation are stored contiguously. This layout
     * is convenient for CPU evaluation, but is not optimal for GPU execution.
     * A GPU implementation should likely use a term-major layout
     * ({@code term × equation}) to allow adjacent threads to access contiguous
     * memory locations.
     *
     * @param equation equation index
     * @param newValues coefficients
     * @param var1Index first variable indices
     * @param var2Index second variable indices
     * @param count number of active terms
     */
    public void setRow(int equation, double[] newValues, int[] var1Index,int[] var2Index, int count) {

        if (count > termsPerEquation)
            throw new IllegalArgumentException(
                    "Too many terms: " + count
            );

        if (var1Index.length < count ||
                var2Index.length < count ||
                newValues.length < count)
            throw new IllegalArgumentException(
                    "Input arrays shorter than count"
            );


        int base = equation * termsPerEquation;//

        for (int i = 0; i < count; i++) {
            values[base + i] = newValues[i];
            this.var1Index[base + i] = var1Index[i];
            this.var2Index[base + i] = var2Index[i];
        }


        // Disable unused terms
        for (int i = count; i < termsPerEquation; i++) {
            values[base + i] = 0.0;

            // safe default indices
            this.var1Index[base + i] = 0;
            this.var2Index[base + i] = 0;
        }
    }

    //There should be a better way of doing it, a loop is bad for something that should be time constant
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

        final double[] values = this.values;
        final int[] var1Index = this.var1Index;
        final int[] var2Index = this.var2Index;

        for (int row = 0; row < equations; row++) {

            double sum = 0.0;

            int base = row * termsPerEquation;
            int end = base + termsPerEquation;

            for (int idx = base; idx < end; idx++) {

                sum += values[idx]
                        * x[var1Index[idx]]
                        * x[var2Index[idx]];
            }

            result[row] = sum;
        }
    }

    @Override
    public void apply(DoubleVector x, DoubleVector result) {
        if (x instanceof CpuDoubleVector xCpu && result instanceof CpuDoubleVector resCpu) {
            multiply(xCpu.array(), resCpu.array());
        }
    }

    //since it represent every possible compination of x_i*x_j. it's size(x) as input and equations as output. It should be able to be non squared.
    @Override
    public int inputSize() {
        return equations;
    }

    @Override
    public int outputSize() {
        return equations;
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

        // Local references so the JIT doesn't re-fetch instance fields on
        // every iteration of the inner loop.
        final double[] values = this.values;
        final int[] var1Index = this.var1Index;
        final int[] var2Index = this.var2Index;
        double sum, c;
        int j, k;

        for (int row = 0; row < equations; row++) {

            sum = 0.0;

            int idx = row * termsPerEquation;
            int end = idx + termsPerEquation;

            while (idx < end) {

                c = values[idx];
                j = var1Index[idx];
                k = var2Index[idx];

                // d(c*x_j*x_k)/dx . direction = c*(x_k*dx_j + x_j*dx_k)
                // This reduces to 2*c*x_j*dx_j automatically when j == k,
                // so no branch is needed to special-case it.
                sum += c * (direction[j] * x[k] + x[j] * direction[k]);
                idx++;
            }

            result[row] = sum;
        }
    }

    public void multiplyJacobian(DoubleVector x, DoubleVector direction, DoubleVector result){
        if (x instanceof CpuDoubleVector xCpu && direction instanceof CpuDoubleVector dirCpu && result instanceof CpuDoubleVector resCpu) {
            multiplyJacobian(xCpu.array(), dirCpu.array(),  resCpu.array());
        }
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


}
