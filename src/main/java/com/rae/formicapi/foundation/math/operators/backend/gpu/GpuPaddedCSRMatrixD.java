package com.rae.formicapi.foundation.math.operators.backend.gpu;

import com.rae.formicapi.foundation.math.operators.linear.MutableMatrix;
import com.rae.formicapi.foundation.math.operators.vectors.IntegerVector;
import com.rae.formicapi.foundation.math.operators.vectors.RealVector;
import org.jetbrains.annotations.Nullable;

import java.util.Arrays;

/**
 * GPU-backed padded CSR matrix. Kernels are static constants defined right
 * here (see {@link PortableKernel}'s class doc for why), bound once in
 * {@link #bindKernels}, dispatched via {@code .use()}/{@code .useBlocking()}
 * -- no more {@code executor.launchCsrMatvec(...)}-style methods on the
 * executor itself.
 */
public class GpuPaddedCSRMatrixD extends GpuExecutable implements MutableMatrix {

    private static final PortableKernel CSR_MATVEC = new PortableKernel("csr_matvec")
            .opencl("""
            #pragma OPENCL EXTENSION cl_khr_fp64 : enable
            __kernel void csr_matvec(__global const double* values, __global const int* colIndex,
                                      const int nnzPerRow, __global const double* x, __global double* result,
                                      const int rows) {
                int row = get_global_id(0);
                if (row >= rows) return;
                double sum = 0.0;
                int base = row * nnzPerRow;
                for (int i = 0; i < nnzPerRow; i++)
                    sum += values[base + i] * x[colIndex[base + i]];
                result[row] = sum;
            }
            """, "cl_khr_fp64")
            .glsl("""
            layout(std430, binding = 0) buffer V { double values[]; };
            layout(std430, binding = 1) buffer C { int colIndex[]; };
            layout(std430, binding = 2) buffer X { double x[]; };
            layout(std430, binding = 3) buffer R { double result[]; };
            layout(push_constant) uniform PC { int gsz; int nnzPerRow; int rows; } pc;
            void main() {
                int row = int(GID);
                if (row >= pc.rows) return;
                double sum = 0.0LF;
                int base = row * pc.nnzPerRow;
                for (int i = 0; i < pc.nnzPerRow; i++)
                    sum += values[base + i] * x[colIndex[base + i]];
                result[row] = sum;
            }
            """, "shaderFloat64");

    // Multiple rows can scatter into the same output column, so this needs
    // an atomic add -- OpenCL has no native fp64 atomic_add, so it's
    // emulated with a compare-and-swap loop over the double's bit pattern
    // (as_ulong/as_double), same trick every OpenCL fp64-atomic-add uses.
    private static final PortableKernel CSR_MATVEC_TRANSPOSE = new PortableKernel("csr_matvec_transpose")
            .opencl("""
            #pragma OPENCL EXTENSION cl_khr_fp64 : enable
            #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
            __kernel void csr_matvec_transpose(__global const double* values, __global const int* colIndex,
                                                const int nnzPerRow, __global const double* x, __global double* result,
                                                const int rows) {
                int row = get_global_id(0);
                if (row >= rows) return;
                double xRow = x[row];
                int base = row * nnzPerRow;
                for (int i = 0; i < nnzPerRow; i++) {
                    int col = colIndex[base + i];
                    double contribution = values[base + i] * xRow;
                    volatile __global ulong* addr = (volatile __global ulong*) &result[col];
                    ulong old;
                    do {
                        old = *addr;
                    } while (atom_cmpxchg(addr, old, as_ulong(as_double(old) + contribution)) != old);
                }
            }
            """, "cl_khr_fp64", "cl_khr_int64_base_atomics")
            .glsl("""
            layout(std430, binding = 0) buffer V { double values[]; };
            layout(std430, binding = 1) buffer C { int colIndex[]; };
            layout(std430, binding = 2) buffer X { double x[]; };
            // result is accumulated into as raw 64-bit patterns so a compare-and-swap
            // loop can emulate the fp64 atomic add that Vulkan core doesn't have.
            layout(std430, binding = 3) buffer R { uint64_t result[]; };
            layout(push_constant) uniform PC { int gsz; int nnzPerRow; int rows; } pc;
            void main() {
                int row = int(GID);
                if (row >= pc.rows) return;
                double xRow = x[row];
                int base = row * pc.nnzPerRow;
                for (int i = 0; i < pc.nnzPerRow; i++) {
                    int col = colIndex[base + i];
                    double contribution = values[base + i] * xRow;
                    uint64_t old = result[col];
                    for (;;) {
                        uint64_t seen = atomicCompSwap(result[col], old, doubleBitsToUint64(uint64BitsToDouble(old) + contribution));
                        if (seen == old) break;
                        old = seen;
                    }
                }
            }
            """, "shaderFloat64", "shaderInt64", "shaderBufferInt64Atomics");

    @Override
    protected void bindKernels(GpuExecutor executor) {
        CSR_MATVEC.bind(executor);
        if (executor.supports(CSR_MATVEC_TRANSPOSE))
            CSR_MATVEC_TRANSPOSE.bind(executor);
    }

    private final int nnzPerRow;

    private int rows;
    private int cols;

    private double[] values;
    private int[] colIndex;

    private @Nullable GpuResource dValues;
    private @Nullable GpuResource dColIndex;

    public GpuPaddedCSRMatrixD(int rows, int cols, int nnzPerRow) {
        if (rows < 0)
            throw new IllegalArgumentException("rows < 0");

        if (cols < 0)
            throw new IllegalArgumentException("cols < 0");

        if (nnzPerRow <= 0)
            throw new IllegalArgumentException("nnzPerRow <= 0");

        this.rows = rows;
        this.cols = cols;
        this.nnzPerRow = nnzPerRow;

        int length = rows * nnzPerRow;

        this.values = new double[length];
        this.colIndex = new int[length];
    }

    @Override
    public void setExecutor(@Nullable GpuExecutor executor) {
        GpuExecutor previous = getExecutor();
        super.setExecutor(executor);

        if (previous != null) {
            if (dValues != null)
                dValues.release();

            if (dColIndex != null)
                dColIndex.release();

            dValues = null;
            dColIndex = null;
        }

        if (executor == null)
            return;

        dValues = executor.allocateDoubleBuffer(Math.max(values.length, 1));
        dColIndex = executor.allocateIntBuffer(Math.max(colIndex.length, 1));

        if (values.length > 0) {
            executor.uploadDoubles(dValues, values, values.length);
            executor.uploadInts(dColIndex, colIndex, colIndex.length);
        }
    }

    @Override
    public void apply(RealVector x, RealVector result) {
        if (!(x instanceof GpuDoubleVector gpuX))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + x.getClass());

        if (!(result instanceof GpuDoubleVector gpuResult))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + result.getClass());

        if (x.size() != cols)
            throw new IllegalArgumentException("x.size() != A.cols()");

        result.resize(rows);

        CSR_MATVEC.use(rows,
                dValues, dColIndex, requireExecutor().scalar(nnzPerRow),
                gpuX.buffer(), gpuResult.buffer(), requireExecutor().scalar(rows));
    }

    @Override
    public void transposeApply(RealVector x, RealVector result) {
        GpuExecutor executor = requireExecutor();

        if (!(x instanceof GpuDoubleVector gpuX))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + x.getClass());

        if (!(result instanceof GpuDoubleVector gpuResult))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + result.getClass());

        if (x.size() != rows)
            throw new IllegalArgumentException("x.size() != A.rows()");

        if (!executor.supports(CSR_MATVEC_TRANSPOSE))
            throw new UnsupportedOperationException(
                    "transposeApply(...) requires 64-bit integer atomics on buffers (OpenCL cl_khr_int64_base_atomics / " +
                            "Vulkan shaderBufferInt64Atomics), which this device doesn't report");

        result.resize(cols);
        gpuResult.clear(); // atomically accumulated into below, so it must start at zero

        // gpuResult's buffer is passed as an arg here, so dispatch's own
        // wait-list building already picks up clear()'s pending event on
        // it -- no explicit ordering needed between the two calls.
        CSR_MATVEC_TRANSPOSE.use(rows,
                dValues, dColIndex, requireExecutor().scalar(nnzPerRow),
                gpuX.buffer(), gpuResult.buffer(), requireExecutor().scalar(rows));
    }

    @Override
    public void add(int row, int col, double value) {
        int index = findIndex(row, col);

        values[index] += value;

        requireExecutor().uploadDoubleAt(dValues, index, values[index]);
    }

    @Override
    public void set(int row, int col, double value) {
        int index = findIndex(row, col);

        values[index] = value;

        requireExecutor().uploadDoubleAt(dValues, index, value);
    }

    @Override
    public double get(int row, int col) {
        checkRow(row);

        int base = row * nnzPerRow;

        for (int i = 0; i < nnzPerRow; i++) {
            if (colIndex[base + i] == col)
                return values[base + i];
        }

        return 0.0;
    }

    /**
     * Gathers the stored entries {@code A[r[i], c[i]]} into a new device
     * vector of size {@code r.size()}; positions outside the sparsity
     * pattern read as {@code 0}, like {@link #get}. Served from the host
     * mirror of the matrix, so it costs one upload of the result.
     */
    @Override
    public RealVector getValues(IntegerVector r, IntegerVector c) {
        if (r.size() != c.size())
            throw new IllegalArgumentException("row and column index vectors differ in size: " + r.size() + " vs " + c.size());

        int[] rows = toArray(r);
        int[] cols = toArray(c);
        double[] out = new double[rows.length];
        for (int i = 0; i < out.length; i++)
            out[i] = get(rows[i], cols[i]);

        return new GpuDoubleVector(requireExecutor(), out);
    }

    private static int[] toArray(IntegerVector v) {
        if (v instanceof GpuIntegerVector gpu)
            return gpu.download(); // one bulk transfer instead of a device round trip per get(i)
        int[] out = new int[v.size()];
        for (int i = 0; i < out.length; i++)
            out[i] = v.get(i);
        return out;
    }

    private int findIndex(int row, int col) {
        checkRow(row);

        int base = row * nnzPerRow;

        for (int i = 0; i < nnzPerRow; i++) {
            if (colIndex[base + i] == col)
                return base + i;
        }

        throw new IllegalArgumentException("CSR entry does not exist: (" + row + ", " + col + ")");
    }

    @Override
    public void setRow(int row, double[] rowValues, int[] rowCols, int count) {
        checkRow(row);

        if (count < 0 || count > nnzPerRow)
            throw new IllegalArgumentException("count out of range");

        if (rowValues.length < count)
            throw new IllegalArgumentException("rowValues too small");

        if (rowCols.length < count)
            throw new IllegalArgumentException("rowCols too small");

        int base = row * nnzPerRow;

        for (int i = 0; i < count; i++) {
            values[base + i] = rowValues[i];
            colIndex[base + i] = rowCols[i];
        }

        /*
         * Unused padded entries are zero.
         */
        for (int i = count; i < nnzPerRow; i++) {
            values[base + i] = 0.0;
            colIndex[base + i] = row < cols ? row : 0;
        }

        requireExecutor().uploadDoubles(dValues, base, values, base, nnzPerRow);
        requireExecutor().uploadInts(dColIndex, base, colIndex, base, nnzPerRow);
    }

    @Override
    public double[] getRowValues(int r) {
        int base = r * nnzPerRow;

        double[] out = new double[nnzPerRow];
        System.arraycopy(values, base, out, 0, nnzPerRow);

        return out;
    }

    @Override
    public int[] getRowCols(int r) {
        int base = r * nnzPerRow;

        int[] out = new int[nnzPerRow];
        System.arraycopy(colIndex, base, out, 0, nnzPerRow);

        return out;
    }

    @Override
    public void resize(int newRows, int newCols) {
        if (newRows < 0)
            throw new IllegalArgumentException("newRows < 0");

        int newLength = newRows * nnzPerRow;

        values = Arrays.copyOf(values, newLength);
        colIndex = Arrays.copyOf(colIndex, newLength);

        rows = newRows;
        cols = newCols;

        GpuExecutor executor = requireExecutor();

        GpuResource newValues = executor.allocateDoubleBuffer(Math.max(newLength, 1));
        GpuResource newColIndex = executor.allocateIntBuffer(Math.max(newLength, 1));

        if (newLength > 0) {
            executor.uploadDoubles(newValues, values, newLength);
            executor.uploadInts(newColIndex, colIndex, newLength);
        }

        if (dValues != null) {
            dValues.release();
        }
        if (dColIndex != null) {
            dColIndex.release();
        }
        dValues = newValues;
        dColIndex = newColIndex;
    }

    @Override
    public void multiply(double[] x, double[] result) {
        throw new UnsupportedOperationException("Use GPU DoubleVector operations instead");
    }

    @Override
    public int rows() {
        return rows;
    }

    @Override
    public int cols() {
        return cols;
    }

    public int nnzPerRow() {
        return nnzPerRow;
    }

    private void checkRow(int row) {
        if (row < 0 || row >= rows)
            throw new IndexOutOfBoundsException("row: " + row);
    }

    public GpuResource valuesBuffer() {
        return dValues;
    }

    public GpuResource colIndexBuffer() {
        return dColIndex;
    }

    public double[] getValues() {
        return values;
    }

    public int[] getColIndex() {
        return colIndex;
    }

    public void close() {
        if (dValues != null) {
            dValues.release();
            dValues = null;
        }

        if (dColIndex != null) {
            dColIndex.release();
            dColIndex = null;
        }
    }
}
