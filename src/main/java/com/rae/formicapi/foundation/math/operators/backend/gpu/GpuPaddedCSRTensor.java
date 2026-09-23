package com.rae.formicapi.foundation.math.operators.backend.gpu;

import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuPaddedCSRTensor;
import com.rae.formicapi.foundation.math.operators.backend.gpu.opencl.OpenCLGpuExecutor;
import com.rae.formicapi.foundation.math.operators.backend.gpu.opencl.OpenCLKernel;
import com.rae.formicapi.foundation.math.operators.backend.gpu.opencl.OpenCLResource;
import com.rae.formicapi.foundation.math.operators.vectors.Vector;
import org.jetbrains.annotations.Nullable;

import java.util.Arrays;

/**
 * GPU backend for {@code PaddedCSRTensor} / {@link CpuPaddedCSRTensor}:
 * {@code F_i(x) = sum(c * x[j1] * x[j2] * ... * x[jN-1])}. Kernels are
 * static constants defined right here, bound once in {@link #bindKernels},
 * dispatched via {@code .use()} -- replacing the
 * {@code executor.launchCsrNApply(...)}-style calls that no longer exist
 * on the executor itself.
 *
 * <p>Same term-major flat {@code int[]} layout for variable indices as
 * before: term {@code idx}'s indices live at
 * {@code [idx*dims, (idx+1)*dims)} where {@code dims = order - 1}.
 */
public class GpuPaddedCSRTensor extends GpuExecutable implements AutoCloseable {

    private static final OpenCLKernel APPLY = new OpenCLKernel("csrn_apply", """
            #pragma OPENCL EXTENSION cl_khr_fp64 : enable
            __kernel void csrn_apply(__global const double* values, __global const int* variableIndices,
                                      const int termsPerEquation, const int dims,
                                      __global const double* x, __global double* result, const int equations) {
                int row = get_global_id(0);
                if (row >= equations) return;
                double sum = 0.0;
                int base = row * termsPerEquation;
                for (int i = 0; i < termsPerEquation; i++) {
                    int idx = base + i;
                    double p = values[idx];
                    for (int d = 0; d < dims; d++)
                        p *= x[variableIndices[idx * dims + d]];
                    sum += p;
                }
                result[row] = sum;
            }
            """, "cl_khr_fp64");

    private static final OpenCLKernel APPLY_JACOBIAN = new OpenCLKernel("csrn_apply_jacobian", """
            #pragma OPENCL EXTENSION cl_khr_fp64 : enable
            __kernel void csrn_apply_jacobian(__global const double* values, __global const int* variableIndices,
                                               const int termsPerEquation, const int dims,
                                               __global const double* x, __global const double* direction,
                                               __global double* result, const int equations) {
                int row = get_global_id(0);
                if (row >= equations) return;
                double sum = 0.0;
                int base = row * termsPerEquation;
                for (int i = 0; i < termsPerEquation; i++) {
                    int idx = base + i;
                    for (int m = 0; m < dims; m++) {
                        double term = values[idx];
                        for (int r = 0; r < dims; r++) {
                            int v = variableIndices[idx * dims + r];
                            term *= (r == m) ? direction[v] : x[v];
                        }
                        sum += term;
                    }
                }
                result[row] = sum;
            }
            """, "cl_khr_fp64");

    @Override
    protected void bindKernels(GpuExecutor executor) {
        APPLY.bind(executor);
        APPLY_JACOBIAN.bind(executor);
    }

    private final int termsPerEquation;
    private final int order;
    private       int equations;

    private double[] values;
    private int[]    variableIndices;

    private @Nullable GpuResource dValues;
    private @Nullable GpuResource dVariableIndices;

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
                dValues.release();
            if (dVariableIndices != null)
                dVariableIndices.release();

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
     * into the host mirror internally).
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

        for (int i = count; i < termsPerEquation; i++) {
            values[base + i] = 0.0;
            Arrays.fill(variableIndices, (base + i) * (order - 1), (base + i + 1) * (order - 1), 0);
        }

        GpuExecutor executor = requireExecutor();
        executor.uploadDoubles(dValues, base, values, base, termsPerEquation);
        executor.uploadInts(dVariableIndices, base * (order - 1), variableIndices, base * (order - 1),
                termsPerEquation * (order - 1));
    }

    /** Adds a coefficient to an existing tensor term, found by a host-side scan then synced back with a single-element upload. */
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
        if (!(x instanceof GpuDoubleVector gpuX))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + x.getClass());

        if (!(result instanceof GpuDoubleVector gpuResult))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + result.getClass());

        result.resize(equations);

        APPLY.use(equations,
                dValues, dVariableIndices, OpenCLResource.of(termsPerEquation), OpenCLResource.of(order - 1),
                gpuX.buffer(), gpuResult.buffer(), OpenCLResource.of(equations));
    }

    /** Evaluates the Jacobian-vector product {@code result = J(x) * direction} on the device. */
    public void applyJacobian(Vector x, Vector direction, Vector result) {
        if (!(x instanceof GpuDoubleVector gpuX))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + x.getClass());

        if (!(direction instanceof GpuDoubleVector gpuDirection))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + direction.getClass());

        if (!(result instanceof GpuDoubleVector gpuResult))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + result.getClass());

        result.resize(equations);

        APPLY_JACOBIAN.use(equations,
                dValues, dVariableIndices, OpenCLResource.of(termsPerEquation), OpenCLResource.of(order - 1),
                gpuX.buffer(), gpuDirection.buffer(), gpuResult.buffer(), OpenCLResource.of(equations));
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

    /** Grows this tensor's equation count in-place; no capacity headroom kept between calls. */
    public void resize(int newEquations) {
        if (newEquations < 0)
            throw new IllegalArgumentException("newEquations < 0");

        int newTerms = newEquations * termsPerEquation;
        int newIndicesLength = newTerms * (order - 1);

        values = Arrays.copyOf(values, newTerms);
        variableIndices = Arrays.copyOf(variableIndices, newIndicesLength);

        equations = newEquations;

        GpuExecutor executor = requireExecutor();

        GpuResource newValuesBuf = executor.allocateDoubleBuffer(Math.max(newTerms, 1));
        GpuResource newIndicesBuf = executor.allocateIntBuffer(Math.max(newIndicesLength, 1));

        if (newTerms > 0) {
            executor.uploadDoubles(newValuesBuf, values, newTerms);
            executor.uploadInts(newIndicesBuf, variableIndices, newIndicesLength);
        }

        dValues.release();
        dVariableIndices.release();

        dValues = newValuesBuf;
        dVariableIndices = newIndicesBuf;
    }

    public double[] getRowValues(int row) {
        checkEquation(row);
        return Arrays.copyOfRange(values, row * termsPerEquation, (row + 1) * termsPerEquation);
    }

    /** Returns row {@code row}'s variable indices as {@code [termsPerEquation][order-1]}. */
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

    public GpuResource valuesBuffer() {
        return dValues;
    }

    public GpuResource variableIndicesBuffer() {
        return dVariableIndices;
    }

    public void close() {
        if (dValues != null) {
            dValues.release();
            dValues = null;
        }
        if (dVariableIndices != null) {
            dVariableIndices.release();
            dVariableIndices = null;
        }
    }
}