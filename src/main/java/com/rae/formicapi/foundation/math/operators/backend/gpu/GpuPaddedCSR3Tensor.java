package com.rae.formicapi.foundation.math.operators.backend.gpu;

import com.rae.formicapi.foundation.math.operators.backend.cpu.CpuPaddedCSR3Tensor;
import com.rae.formicapi.foundation.math.operators.backend.gpu.opencl.OpenCLGpuExecutor;
import com.rae.formicapi.foundation.math.operators.backend.gpu.opencl.OpenCLKernel;
import com.rae.formicapi.foundation.math.operators.backend.gpu.opencl.OpenCLResource;
import com.rae.formicapi.foundation.math.operators.vectors.Vector;
import org.jetbrains.annotations.Nullable;

import java.util.Arrays;

/**
 * GPU backend for {@code PaddedCSR2Tensor} / {@link CpuPaddedCSR3Tensor}:
 * {@code F_i(x) = sum(c[i,j,k] * x[j] * x[k])}. Kernels are static
 * constants defined right here (see {@link OpenCLKernel}'s class doc),
 * bound once in {@link #bindKernels}, dispatched via {@code .use()} --
 * replacing the {@code executor.launchCsr2Apply(...)}-style calls that no
 * longer exist on the executor itself.
 */
public class GpuPaddedCSR3Tensor extends GpuExecutable implements AutoCloseable {

    private static final OpenCLKernel APPLY = new OpenCLKernel("csr2_apply", """
            #pragma OPENCL EXTENSION cl_khr_fp64 : enable
            __kernel void csr2_apply(__global const double* values, __global const int* var1Index,
                                      __global const int* var2Index, const int termsPerEquation,
                                      __global const double* x, __global double* result, const int equations) {
                int row = get_global_id(0);
                if (row >= equations) return;
                double sum = 0.0;
                int base = row * termsPerEquation;
                for (int i = 0; i < termsPerEquation; i++) {
                    int idx = base + i;
                    sum += values[idx] * x[var1Index[idx]] * x[var2Index[idx]];
                }
                result[row] = sum;
            }
            """, "cl_khr_fp64");

    private static final OpenCLKernel APPLY_JACOBIAN = new OpenCLKernel("csr2_apply_jacobian", """
            #pragma OPENCL EXTENSION cl_khr_fp64 : enable
            __kernel void csr2_apply_jacobian(__global const double* values, __global const int* var1Index,
                                               __global const int* var2Index, const int termsPerEquation,
                                               __global const double* x, __global const double* direction,
                                               __global double* result, const int equations) {
                int row = get_global_id(0);
                if (row >= equations) return;
                double sum = 0.0;
                int base = row * termsPerEquation;
                for (int i = 0; i < termsPerEquation; i++) {
                    int idx = base + i;
                    double c = values[idx];
                    int j = var1Index[idx];
                    int k = var2Index[idx];
                    // d(c*x_j*x_k)/dx . direction = c*(x_k*dx_j + x_j*dx_k);
                    // reduces to 2*c*x_j*dx_j automatically when j == k.
                    sum += c * (direction[j] * x[k] + x[j] * direction[k]);
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
    private       int equations;

    private double[] values;
    private int[]    var1Index;
    private int[]    var2Index;

    private @Nullable GpuResource dValues;
    private @Nullable GpuResource dVar1Index;
    private @Nullable GpuResource dVar2Index;

    public GpuPaddedCSR3Tensor(int equations, int termsPerEquation) {
        if (equations < 0)
            throw new IllegalArgumentException("equations < 0");

        if (termsPerEquation <= 0)
            throw new IllegalArgumentException("termsPerEquation <= 0");

        this.equations = equations;
        this.termsPerEquation = termsPerEquation;

        int length = equations * termsPerEquation;

        this.values = new double[length];
        this.var1Index = new int[length];
        this.var2Index = new int[length];
    }

    @Override
    public void setExecutor(@Nullable GpuExecutor executor) {
        GpuExecutor previous = getExecutor();
        super.setExecutor(executor);

        if (previous != null) {
            if (dValues != null)
                dValues.release();
            if (dVar1Index != null)
                dVar1Index.release();
            if (dVar2Index != null)
                dVar2Index.release();

            dValues = null;
            dVar1Index = null;
            dVar2Index = null;
        }

        if (executor == null)
            return;

        dValues = executor.allocateDoubleBuffer(Math.max(values.length, 1));
        dVar1Index = executor.allocateIntBuffer(Math.max(var1Index.length, 1));
        dVar2Index = executor.allocateIntBuffer(Math.max(var2Index.length, 1));

        if (values.length > 0) {
            executor.uploadDoubles(dValues, values, values.length);
            executor.uploadInts(dVar1Index, var1Index, var1Index.length);
            executor.uploadInts(dVar2Index, var2Index, var2Index.length);
        }
    }

    /**
     * Replaces one equation's quadratic structure. See
     * {@link CpuPaddedCSR3Tensor#setRow} for the full contract; behaves
     * identically here, plus syncing the row to the device buffer.
     */
    public void setRow(int equation, double[] newValues, int[] newVar1Index, int[] newVar2Index, int count) {
        checkEquation(equation);

        if (count < 0 || count > termsPerEquation)
            throw new IllegalArgumentException("count out of range");

        if (newValues.length < count || newVar1Index.length < count || newVar2Index.length < count)
            throw new IllegalArgumentException("input arrays shorter than count");

        int base = equation * termsPerEquation;

        for (int i = 0; i < count; i++) {
            values[base + i] = newValues[i];
            var1Index[base + i] = newVar1Index[i];
            var2Index[base + i] = newVar2Index[i];
        }

        for (int i = count; i < termsPerEquation; i++) {
            values[base + i] = 0.0;
            var1Index[base + i] = 0;
            var2Index[base + i] = 0;
        }

        GpuExecutor executor = requireExecutor();
        executor.uploadDoubles(dValues, base, values, base, termsPerEquation);
        executor.uploadInts(dVar1Index, base, var1Index, base, termsPerEquation);
        executor.uploadInts(dVar2Index, base, var2Index, base, termsPerEquation);
    }

    /**
     * Adds a coefficient to an existing quadratic term, found by a host-side
     * scan (no device round-trip) then synced back with a single-element
     * upload.
     *
     * @throws IllegalStateException if the term does not exist
     */
    public void add(int equation, int var1, int var2, double value) {
        if (value == 0.0)
            return;

        int index = findIndex(equation, var1, var2);
        values[index] += value;

        requireExecutor().uploadDoubleAt(dValues, index, values[index]);
    }

    private int findIndex(int equation, int var1, int var2) {
        checkEquation(equation);

        int base = equation * termsPerEquation;

        for (int i = 0; i < termsPerEquation; i++) {
            int idx = base + i;
            if (var1Index[idx] == var1 && var2Index[idx] == var2)
                return idx;
        }

        throw new IllegalStateException("Tensor entry does not exist: (" + equation + "," + var1 + "," + var2 + ")");
    }

    /** Evaluates {@code result = F(x)} on the device. {@code result} is resized to {@link #equations()} first. */
    public void apply(Vector x, Vector result) {
        if (!(x instanceof GpuDoubleVector gpuX))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + x.getClass());

        if (!(result instanceof GpuDoubleVector gpuResult))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + result.getClass());

        result.resize(equations);

        APPLY.use(equations,
                dValues, dVar1Index, dVar2Index, OpenCLResource.of(termsPerEquation),
                gpuX.buffer(), gpuResult.buffer(), OpenCLResource.of(equations));
    }

    /**
     * Evaluates the Jacobian-vector product {@code result = J(x) * direction}
     * on the device. See {@link CpuPaddedCSR3Tensor#multiplyJacobian} for the
     * math (handles {@code j == k} correctly without a special case).
     */
    public void applyJacobian(Vector x, Vector direction, Vector result) {
        if (!(x instanceof GpuDoubleVector gpuX))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + x.getClass());

        if (!(direction instanceof GpuDoubleVector gpuDirection))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + direction.getClass());

        if (!(result instanceof GpuDoubleVector gpuResult))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + result.getClass());

        result.resize(equations);

        APPLY_JACOBIAN.use(equations,
                dValues, dVar1Index, dVar2Index, OpenCLResource.of(termsPerEquation),
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

    /**
     * Grows this tensor's equation count in-place, reallocating and
     * re-uploading the device buffers to match - no capacity headroom kept
     * between calls.
     */
    public void resize(int newEquations) {
        if (newEquations < 0)
            throw new IllegalArgumentException("newEquations < 0");

        int newLength = newEquations * termsPerEquation;

        values = Arrays.copyOf(values, newLength);
        var1Index = Arrays.copyOf(var1Index, newLength);
        var2Index = Arrays.copyOf(var2Index, newLength);

        equations = newEquations;

        GpuExecutor executor = requireExecutor();

        GpuResource newValuesBuf = executor.allocateDoubleBuffer(Math.max(newLength, 1));
        GpuResource newVar1Buf = executor.allocateIntBuffer(Math.max(newLength, 1));
        GpuResource newVar2Buf = executor.allocateIntBuffer(Math.max(newLength, 1));

        if (newLength > 0) {
            executor.uploadDoubles(newValuesBuf, values, newLength);
            executor.uploadInts(newVar1Buf, var1Index, newLength);
            executor.uploadInts(newVar2Buf, var2Index, newLength);
        }

        dValues.release();
        dVar1Index.release();
        dVar2Index.release();

        dValues = newValuesBuf;
        dVar1Index = newVar1Buf;
        dVar2Index = newVar2Buf;
    }

    public double[] getRowValues(int row) {
        checkEquation(row);
        return Arrays.copyOfRange(values, row * termsPerEquation, (row + 1) * termsPerEquation);
    }

    public int[] getRowVar1(int row) {
        checkEquation(row);
        return Arrays.copyOfRange(var1Index, row * termsPerEquation, (row + 1) * termsPerEquation);
    }

    public int[] getRowVar2(int row) {
        checkEquation(row);
        return Arrays.copyOfRange(var2Index, row * termsPerEquation, (row + 1) * termsPerEquation);
    }

    private void checkEquation(int equation) {
        if (equation < 0 || equation >= equations)
            throw new IndexOutOfBoundsException("equation: " + equation);
    }

    public @Nullable GpuResource valuesBuffer() {
        return dValues;
    }

    public @Nullable GpuResource var1IndexBuffer() {
        return dVar1Index;
    }

    public @Nullable GpuResource var2IndexBuffer() {
        return dVar2Index;
    }

    public void close() {
        if (dValues != null) {
            dValues.release();
            dValues = null;
        }
        if (dVar1Index != null) {
            dVar1Index.release();
            dVar1Index = null;
        }
        if (dVar2Index != null) {
            dVar2Index.release();
            dVar2Index = null;
        }
    }
}