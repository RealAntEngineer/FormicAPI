package com.rae.formicapi.foundation.math.operators.backend.gpu;

import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuPaddedCSRTensor;
import com.rae.formicapi.foundation.math.operators.vectors.Vector;
import org.jetbrains.annotations.Nullable;
import org.jocl.cl_mem;

import java.util.Arrays;

/**
 * GPU backend for {@code PaddedCSRTensor} / {@link CpuPaddedCSRTensor}.
 *
 * <p>Fixed-structure sparse N-order tensor:
 *
 * <pre>
 * F_i(x) = sum(c * x[j1] * x[j2] * ... * x[jN-1])
 * </pre>
 *
 * <p>Same host-mirror + lazily-allocated device buffer pattern as
 * {@link GpuPaddedCSRMatrix} and {@link GpuPaddedCSR3Tensor}. One layout
 * difference from {@link CpuPaddedCSRTensor}: variable indices are kept as
 * a single flat {@code int[]} (term-major: term {@code idx}'s indices live
 * at {@code [idx*(order-1), (idx+1)*(order-1))}) rather than a jagged
 * {@code int[][]} - a jagged array has no single-buffer device
 * representation anyway, so the host mirror uses the same flattened shape
 * the device buffer needs, instead of flattening on every sync.
 */
public class GpuPaddedCSRTensor extends GpuExecutable {

    private final int termsPerEquation;
    private final int order;
    private       int equations;

    private double[] values;
    private int[]    variableIndices;

    private cl_mem dValues;
    private cl_mem dVariableIndices;

    public GpuPaddedCSRTensor(int order, int equations, int termsPerEquation) {
        if (order < 2)
            throw new IllegalArgumentException("Order must be > 1");

        if (equations < 0)
            throw new IllegalArgumentException("equations < 0");

        if (termsPerEquation <= 0)
            throw new IllegalArgumentException("termsPerEquation <= 0");

        this.order = order;
        this.equations = equations;
        this.termsPerEquation = termsPerEquation;

        int terms = equations * termsPerEquation;

        this.values = new double[terms];
        this.variableIndices = new int[terms * (order - 1)];
    }

    @Override
    public void setExecutor(@Nullable GpuExecutor executor) {
        GpuExecutor previous = getExecutor();
        super.setExecutor(executor);

        if (previous != null) {
            if (dValues != null)
                previous.release(dValues);
            if (dVariableIndices != null)
                previous.release(dVariableIndices);

            dValues = null;
            dVariableIndices = null;
        }

        if (executor == null)
            return;

        dValues = executor.allocateDoubleBuffer(Math.max(values.length, 1));
        dVariableIndices = executor.allocateIntBuffer(Math.max(variableIndices.length, 1));

        if (values.length > 0) {
            executor.uploadDoubles(dValues, values, values.length);
            executor.uploadInts(dVariableIndices, variableIndices, variableIndices.length);
        }
    }

    /**
     * Overwrites one equation's structure and coefficients. See
     * {@link CpuPaddedCSRTensor#setRow} for the full contract;
     * {@code varIndices} is still {@code [count][order-1]} here (flattened
     * into the host mirror internally), matching the Cpu version's caller-
     * facing shape even though storage differs.
     */
    public void setRow(int equation, double[] newValues, int[][] varIndices, int count) {
        checkEquation(equation);

        if (count < 0 || count > termsPerEquation)
            throw new IllegalArgumentException("count out of range");

        if (newValues.length < count)
            throw new IllegalArgumentException("newValues shorter than count");

        for (int i = 0; i < count; i++)
            if (varIndices[i].length != order - 1)
                throw new IllegalArgumentException("Wrong tensor order");

        int base = equation * termsPerEquation;

        for (int i = 0; i < count; i++) {
            values[base + i] = newValues[i];
            System.arraycopy(varIndices[i], 0, variableIndices, (base + i) * (order - 1), order - 1);
        }

        // Disable unused terms - zero value, indices 0 (always in-bounds).
        for (int i = count; i < termsPerEquation; i++) {
            values[base + i] = 0.0;
            Arrays.fill(variableIndices, (base + i) * (order - 1), (base + i + 1) * (order - 1), 0);
        }

        GpuExecutor executor = requireExecutor();
        executor.uploadDoubles(dValues, base, values, base, termsPerEquation);
        executor.uploadInts(dVariableIndices, base * (order - 1), variableIndices, base * (order - 1),
                termsPerEquation * (order - 1));
        requireExecutor().finish();
    }

    /**
     * Adds a coefficient to an existing tensor term, found by a host-side
     * scan then synced back with a single-element upload.
     */
    public void add(int equation, double value, int... vars) {
        if (value == 0.0)
            return;

        if (vars.length != order - 1)
            throw new IllegalArgumentException("Wrong tensor order");

        int index = findIndex(equation, vars);
        values[index] += value;

        requireExecutor().uploadDoubleAt(dValues, index, values[index]);
    }

    private int findIndex(int equation, int[] vars) {
        checkEquation(equation);

        int base = equation * termsPerEquation;
        int dims = order - 1;

        for (int i = 0; i < termsPerEquation; i++) {
            int idx = base + i;
            int vbase = idx * dims;
            if (Arrays.equals(variableIndices, vbase, vbase + dims, vars, 0, dims))
                return idx;
        }

        throw new IllegalStateException("Tensor entry does not exist");
    }

    /** Evaluates {@code result = F(x)} on the device. {@code result} is resized to {@link #equations()} first. */
    public void apply(Vector x, Vector result) {
        GpuExecutor executor = requireExecutor();

        if (!(x instanceof GpuDoubleVector gpuX))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + x.getClass());

        if (!(result instanceof GpuDoubleVector gpuResult))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + result.getClass());

        result.resize(equations);

        executor.launchCsrNApply(dValues, dVariableIndices, termsPerEquation, order - 1,
                gpuX.buffer(), gpuResult.buffer(), equations);
        requireExecutor().finish();
    }

    /** Evaluates the Jacobian-vector product {@code result = J(x) * direction} on the device. */
    public void applyJacobian(Vector x, Vector direction, Vector result) {
        GpuExecutor executor = requireExecutor();

        if (!(x instanceof GpuDoubleVector gpuX))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + x.getClass());

        if (!(direction instanceof GpuDoubleVector gpuDirection))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + direction.getClass());

        if (!(result instanceof GpuDoubleVector gpuResult))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + result.getClass());

        result.resize(equations);

        executor.launchCsrNApplyJacobian(dValues, dVariableIndices, termsPerEquation, order - 1,
                gpuX.buffer(), gpuDirection.buffer(), gpuResult.buffer(), equations);
        requireExecutor().finish();
    }

    public void multiply(double[] x, double[] result) {
        throw new UnsupportedOperationException("Use GPU DoubleVector operations instead");
    }

    public void multiplyJacobian(double[] x, double[] direction, double[] result) {
        throw new UnsupportedOperationException("Use GPU DoubleVector operations instead");
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

    /** Grows this tensor's equation count in-place; same always-reallocate style as {@link GpuPaddedCSRMatrix#resize}. */
    public void resize(int newEquations) {
        if (newEquations < 0)
            throw new IllegalArgumentException("newEquations < 0");

        int newTerms = newEquations * termsPerEquation;
        int newIndicesLength = newTerms * (order - 1);

        values = Arrays.copyOf(values, newTerms);
        variableIndices = Arrays.copyOf(variableIndices, newIndicesLength);

        equations = newEquations;

        GpuExecutor executor = requireExecutor();

        cl_mem newValuesBuf = executor.allocateDoubleBuffer(Math.max(newTerms, 1));
        cl_mem newIndicesBuf = executor.allocateIntBuffer(Math.max(newIndicesLength, 1));

        if (newTerms > 0) {
            executor.uploadDoubles(newValuesBuf, values, newTerms);
            executor.uploadInts(newIndicesBuf, variableIndices, newIndicesLength);
        }

        executor.release(dValues);
        executor.release(dVariableIndices);

        dValues = newValuesBuf;
        dVariableIndices = newIndicesBuf;
    }

    public double[] getRowValues(int row) {
        checkEquation(row);
        return Arrays.copyOfRange(values, row * termsPerEquation, (row + 1) * termsPerEquation);
    }

    /** Returns row {@code row}'s variable indices as {@code [termsPerEquation][order-1]}, matching {@link CpuPaddedCSRTensor}'s caller-facing shape. */
    public int[][] getRowVarIndices(int row) {
        checkEquation(row);

        int dims = order - 1;
        int[][] out = new int[termsPerEquation][dims];
        int base = row * termsPerEquation * dims;

        for (int i = 0; i < termsPerEquation; i++)
            System.arraycopy(variableIndices, base + i * dims, out[i], 0, dims);

        return out;
    }

    private void checkEquation(int equation) {
        if (equation < 0 || equation >= equations)
            throw new IndexOutOfBoundsException("equation: " + equation);
    }

    public cl_mem valuesBuffer() {
        return dValues;
    }

    public cl_mem variableIndicesBuffer() {
        return dVariableIndices;
    }

    public void close() {
        GpuExecutor executor = getExecutor();
        if (executor == null)
            return;

        if (dValues != null) {
            executor.release(dValues);
            dValues = null;
        }
        if (dVariableIndices != null) {
            executor.release(dVariableIndices);
            dVariableIndices = null;
        }
    }
}