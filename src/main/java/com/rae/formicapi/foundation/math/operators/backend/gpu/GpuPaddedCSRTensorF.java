package com.rae.formicapi.foundation.math.operators.backend.gpu;

import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuPaddedCSRTensorD;
import com.rae.formicapi.foundation.math.operators.nonlinear.PolynomialOperator;
import com.rae.formicapi.foundation.math.operators.vectors.RealVector;
import org.jetbrains.annotations.Nullable;

import java.util.Arrays;

/**
 * Single-precision twin of {@link GpuPaddedCSRTensorD}:
 * {@code F_i(x) = sum(c * x[j1] * x[j2] * ... * x[jN-1])}, with {@code float}
 * storage and kernels that need no fp64, working on {@link GpuFloatVector}s.
 * The public API stays {@code double}-typed like {@link GpuFloatVector}'s
 * (values are rounded to {@code float} on the way in), so call sites can swap
 * the D and F classes freely.
 *
 * <p>Same term-major flat {@code int[]} layout for variable indices as the D
 * class: term {@code idx}'s indices live at {@code [idx*dims, (idx+1)*dims)}
 * where {@code dims = order - 1}.
 */
public class GpuPaddedCSRTensorF extends GpuExecutable implements AutoCloseable, PolynomialOperator {

    private static final PortableKernel APPLY = new PortableKernel("csrn_apply_f")
            .opencl("""
                    __kernel void csrn_apply_f(__global const float* values, __global const int* variableIndices,
                                                const int termsPerEquation, const int dims,
                                                __global const float* x, __global float* result, const int equations) {
                        int row = get_global_id(0);
                        if (row >= equations) return;
                        float sum = 0.0f;
                        int base = row * termsPerEquation;
                        for (int i = 0; i < termsPerEquation; i++) {
                            int idx = base + i;
                            float p = values[idx];
                            for (int d = 0; d < dims; d++)
                                p *= x[variableIndices[idx * dims + d]];
                            sum += p;
                        }
                        result[row] = sum;
                    }
                    """)
            .glsl("""
                    layout(std430, binding = 0) buffer V { float values[]; };
                    layout(std430, binding = 1) buffer I { int variableIndices[]; };
                    layout(std430, binding = 2) buffer X { float x[]; };
                    layout(std430, binding = 3) buffer R { float result[]; };
                    layout(push_constant) uniform PC { int gsz; int termsPerEquation; int dims; int equations; } pc;
                    void main() {
                        int row = int(GID);
                        if (row >= pc.equations) return;
                        float sum = 0.0;
                        int base = row * pc.termsPerEquation;
                        for (int i = 0; i < pc.termsPerEquation; i++) {
                            int idx = base + i;
                            float p = values[idx];
                            for (int d = 0; d < pc.dims; d++)
                                p *= x[variableIndices[idx * pc.dims + d]];
                            sum += p;
                        }
                        result[row] = sum;
                    }
                    """, "shaderFloat32");

    private static final PortableKernel APPLY_JACOBIAN = new PortableKernel("csrn_apply_jacobian_f")
            .opencl("""
                    __kernel void csrn_apply_jacobian_f(__global const float* values, __global const int* variableIndices,
                                                         const int termsPerEquation, const int dims,
                                                         __global const float* x, __global const float* direction,
                                                         __global float* result, const int equations) {
                        int row = get_global_id(0);
                        if (row >= equations) return;
                        float sum = 0.0f;
                        int base = row * termsPerEquation;
                        for (int i = 0; i < termsPerEquation; i++) {
                            int idx = base + i;
                            for (int m = 0; m < dims; m++) {
                                float term = values[idx];
                                for (int r = 0; r < dims; r++) {
                                    int v = variableIndices[idx * dims + r];
                                    term *= (r == m) ? direction[v] : x[v];
                                }
                                sum += term;
                            }
                        }
                        result[row] = sum;
                    }
                    """)
            .glsl("""
                    layout(std430, binding = 0) buffer V { float values[]; };
                    layout(std430, binding = 1) buffer I { int variableIndices[]; };
                    layout(std430, binding = 2) buffer X { float x[]; };
                    layout(std430, binding = 3) buffer Dir { float direction[]; };
                    layout(std430, binding = 4) buffer R { float result[]; };
                    layout(push_constant) uniform PC { int gsz; int termsPerEquation; int dims; int equations; } pc;
                    void main() {
                        int row = int(GID);
                        if (row >= pc.equations) return;
                        float sum = 0.0;
                        int base = row * pc.termsPerEquation;
                        for (int i = 0; i < pc.termsPerEquation; i++) {
                            int idx = base + i;
                            for (int m = 0; m < pc.dims; m++) {
                                float term = values[idx];
                                for (int r = 0; r < pc.dims; r++) {
                                    int v = variableIndices[idx * pc.dims + r];
                                    term *= (r == m) ? direction[v] : x[v];
                                }
                                sum += term;
                            }
                        }
                        result[row] = sum;
                    }
                    """, "shaderFloat32");
    private final int termsPerEquation;
    private final int order;
    private       int equations;
    private float[] values;
    private int[]   variableIndices;
    private @Nullable GpuResource dValues;
    private @Nullable GpuResource dVariableIndices;
    public GpuPaddedCSRTensorF(int order, int equations, int termsPerEquation) {
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

        this.values = new float[terms];
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

        dValues = executor.allocateFloatBuffer(Math.max(values.length, 1));
        dVariableIndices = executor.allocateIntBuffer(Math.max(variableIndices.length, 1));

        if (values.length > 0) {
            executor.uploadFloats(dValues, values, values.length);
            executor.uploadInts(dVariableIndices, variableIndices, variableIndices.length);
        }
    }

    @Override
    protected void bindKernels(GpuExecutor executor) {
        APPLY.bind(executor);
        APPLY_JACOBIAN.bind(executor);
    }

    /**
     * Overwrites one equation's structure and coefficients. See
     * {@link CpuPaddedCSRTensorD#setRow} for the full contract;
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
            values[base + i] = (float) newValues[i];
            System.arraycopy(varIndices[i], 0, variableIndices, (base + i) * (order - 1), order - 1);
        }

        for (int i = count; i < termsPerEquation; i++) {
            values[base + i] = 0.0f;
            Arrays.fill(variableIndices, (base + i) * (order - 1), (base + i + 1) * (order - 1), 0);
        }

        GpuExecutor executor = requireExecutor();
        executor.uploadFloats(dValues, base, values, base, termsPerEquation);
        executor.uploadInts(dVariableIndices, base * (order - 1), variableIndices, base * (order - 1),
                termsPerEquation * (order - 1));
    }

    private void checkEquation(int equation) {
        if (equation < 0 || equation >= equations)
            throw new IndexOutOfBoundsException("equation: " + equation);
    }

    /**
     * Adds a coefficient to an existing tensor term, found by a host-side scan then synced back with a single-element upload.
     */
    public void add(int equation, double value, int... vars) {
        if (value == 0.0)
            return;

        if (vars.length != order - 1)
            throw new IllegalArgumentException("Wrong tensor order");

        int index = findIndex(equation, vars);
        values[index] = (float) (values[index] + value); // one rounding, from the (near-)exact sum

        requireExecutor().uploadFloatAt(dValues, index, values[index]);
    }

    private int findIndex(int equation, int[] vars) {
        checkEquation(equation);

        int base = equation * termsPerEquation;
        int dims = order - 1;

        for (int i = 0; i < termsPerEquation; i++) {
            int idx   = base + i;
            int vbase = idx * dims;
            if (Arrays.equals(variableIndices, vbase, vbase + dims, vars, 0, dims))
                return idx;
        }

        throw new IllegalStateException("Tensor entry does not exist");
    }

    /**
     * Evaluates {@code result = F(x)} on the device. {@code result} is resized to {@link #equations()} first.
     */
    public void apply(RealVector x, RealVector result) {
        if (!(x instanceof GpuFloatVector gpuX))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + x.getClass());

        if (!(result instanceof GpuFloatVector gpuResult))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + result.getClass());

        result.resize(equations);

        APPLY.use(equations,
                dValues, dVariableIndices, requireExecutor().scalar(termsPerEquation), requireExecutor().scalar(order - 1),
                gpuX.buffer(), gpuResult.buffer(), requireExecutor().scalar(equations));
    }

    @Override
    public int inputSize() {
        return equations;
    }

    @Override
    public int outputSize() {
        return equations;
    }

    /**
     * Evaluates the Jacobian-vector product {@code result = J(x) * direction} on the device.
     */
    @Override
    public void multiplyJacobian(RealVector x, RealVector direction, RealVector result) {
        if (!(x instanceof GpuFloatVector gpuX))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + x.getClass());

        if (!(direction instanceof GpuFloatVector gpuDirection))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + direction.getClass());

        if (!(result instanceof GpuFloatVector gpuResult))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + result.getClass());

        result.resize(equations);

        APPLY_JACOBIAN.use(equations,
                dValues, dVariableIndices, requireExecutor().scalar(termsPerEquation), requireExecutor().scalar(order - 1),
                gpuX.buffer(), gpuDirection.buffer(), gpuResult.buffer(), requireExecutor().scalar(equations));
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

    /**
     * Grows this tensor's equation count in-place; no capacity headroom kept between calls.
     */
    public void resize(int newEquations) {
        if (newEquations < 0)
            throw new IllegalArgumentException("newEquations < 0");

        int newTerms         = newEquations * termsPerEquation;
        int newIndicesLength = newTerms * (order - 1);

        values = Arrays.copyOf(values, newTerms);
        variableIndices = Arrays.copyOf(variableIndices, newIndicesLength);

        equations = newEquations;

        GpuExecutor executor = requireExecutor();

        GpuResource newValuesBuf  = executor.allocateFloatBuffer(Math.max(newTerms, 1));
        GpuResource newIndicesBuf = executor.allocateIntBuffer(Math.max(newIndicesLength, 1));

        if (newTerms > 0) {
            executor.uploadFloats(newValuesBuf, values, newTerms);
            executor.uploadInts(newIndicesBuf, variableIndices, newIndicesLength);
        }

        if (dValues != null)
            dValues.release();
        if (dVariableIndices != null)
            dVariableIndices.release();

        dValues = newValuesBuf;
        dVariableIndices = newIndicesBuf;
    }

    public double[] getRowValues(int row) {
        checkEquation(row);
        double[] out  = new double[termsPerEquation];
        int      base = row * termsPerEquation;
        for (int i = 0; i < termsPerEquation; i++)
            out[i] = values[base + i];
        return out;
    }

    /**
     * Returns row {@code row}'s variable indices as {@code [termsPerEquation][order-1]}.
     */
    public int[][] getRowVarIndices(int row) {
        checkEquation(row);

        int     dims = order - 1;
        int[][] out  = new int[termsPerEquation][dims];
        int     base = row * termsPerEquation * dims;

        for (int i = 0; i < termsPerEquation; i++)
            System.arraycopy(variableIndices, base + i * dims, out[i], 0, dims);

        return out;
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