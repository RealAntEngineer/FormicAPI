package com.rae.formicapi.foundation.math.operators.backend.gpu;

import com.rae.formicapi.foundation.math.operators.linear.MutableMatrix;
import com.rae.formicapi.foundation.math.operators.vectors.IntegerVector;
import com.rae.formicapi.foundation.math.operators.vectors.RealVector;
import org.jetbrains.annotations.Nullable;

import java.util.Arrays;

/**
 * Single-precision twin of {@link GpuPaddedCSRMatrixD}: same padded-CSR
 * layout and API, but the values live as {@code float} on the host mirror and
 * on the device, and the kernels need no fp64 and no atomics, so it runs on
 * every Vulkan/OpenCL device. It pairs with {@link GpuFloatVector}.
 *
 * <p>Like {@link GpuFloatVector}, the public API stays {@code double}-typed
 * (it implements the precision-agnostic {@link MutableMatrix}); values are
 * rounded to {@code float} on the way in and widened on the way out.
 *
 * <p><b>{@link #transposeApply}</b> is an atomic-free gather: every output
 * column sums the entries that point at it, using a column -> entries index
 * built on the host and cached (see {@link #ensureTransposeIndex}). Only the
 * sparsity pattern ({@code setRow}/{@code resize}) invalidates it;
 * {@code set}/{@code add} just change values, which the kernel reads by flat
 * index. The summation order is fixed, so results are reproducible run to run.
 */
public class GpuPaddedCSRMatrixF extends GpuExecutable implements MutableMatrix {

    private static final PortableKernel CSR_MATVEC = new PortableKernel("csr_matvec_f")
            .opencl("""
            __kernel void csr_matvec_f(__global const float* values, __global const int* colIndex,
                                        const int nnzPerRow, __global const float* x, __global float* result,
                                        const int rows) {
                int row = get_global_id(0);
                if (row >= rows) return;
                float sum = 0.0f;
                int base = row * nnzPerRow;
                for (int i = 0; i < nnzPerRow; i++)
                    sum += values[base + i] * x[colIndex[base + i]];
                result[row] = sum;
            }
            """)
            .glsl("""
            layout(std430, binding = 0) buffer V { float values[]; };
            layout(std430, binding = 1) buffer C { int colIndex[]; };
            layout(std430, binding = 2) buffer X { float x[]; };
            layout(std430, binding = 3) buffer R { float result[]; };
            layout(push_constant) uniform PC { int gsz; int nnzPerRow; int rows; } pc;
            void main() {
                int row = int(GID);
                if (row >= pc.rows) return;
                float sum = 0.0;
                int base = row * pc.nnzPerRow;
                for (int i = 0; i < pc.nnzPerRow; i++)
                    sum += values[base + i] * x[colIndex[base + i]];
                result[row] = sum;
            }
            """, "shaderFloat32");

    // One thread per output column; transPtr/transEntry are CSR-style over flat indices into `values`.
    private static final PortableKernel CSR_MATVEC_TRANSPOSE = new PortableKernel("csr_matvec_transpose_f")
            .opencl("""
            __kernel void csr_matvec_transpose_f(__global const float* values, __global const int* transPtr,
                                                  __global const int* transEntry, __global const float* x,
                                                  __global float* result, const int nnzPerRow, const int cols) {
                int col = get_global_id(0);
                if (col >= cols) return;
                float sum = 0.0f;
                for (int k = transPtr[col]; k < transPtr[col + 1]; k++) {
                    int e = transEntry[k];
                    sum += values[e] * x[e / nnzPerRow];
                }
                result[col] = sum;
            }
            """)
            .glsl("""
            layout(std430, binding = 0) buffer V { float values[]; };
            layout(std430, binding = 1) buffer P { int transPtr[]; };
            layout(std430, binding = 2) buffer E { int transEntry[]; };
            layout(std430, binding = 3) buffer X { float x[]; };
            layout(std430, binding = 4) buffer R { float result[]; };
            layout(push_constant) uniform PC { int gsz; int nnzPerRow; int cols; } pc;
            void main() {
                int col = int(GID);
                if (col >= pc.cols) return;
                float sum = 0.0;
                for (int k = transPtr[col]; k < transPtr[col + 1]; k++) {
                    int e = transEntry[k];
                    sum += values[e] * x[e / pc.nnzPerRow];
                }
                result[col] = sum;
            }
            """, "shaderFloat32");

    @Override
    protected void bindKernels(GpuExecutor executor) {
        CSR_MATVEC.bind(executor);
        CSR_MATVEC_TRANSPOSE.bind(executor);
    }

    private final int nnzPerRow;

    private int rows;
    private int cols;

    private float[] values;
    private int[] colIndex;

    private @Nullable GpuResource dValues;
    private @Nullable GpuResource dColIndex;

    // Column -> entries index for transposeApply. Built lazily; only the sparsity pattern (colIndex) invalidates it.
    private @Nullable GpuResource dTransPtr;
    private @Nullable GpuResource dTransEntry;
    private boolean transposeDirty = true;

    public GpuPaddedCSRMatrixF(int rows, int cols, int nnzPerRow) {
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

        this.values = new float[length];
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
            releaseTransposeIndex();
        }

        if (executor == null)
            return;

        dValues = executor.allocateFloatBuffer(Math.max(values.length, 1));
        dColIndex = executor.allocateIntBuffer(Math.max(colIndex.length, 1));

        if (values.length > 0) {
            executor.uploadFloats(dValues, values, values.length);
            executor.uploadInts(dColIndex, colIndex, colIndex.length);
        }
    }

    @Override
    public void apply(RealVector x, RealVector result) {
        if (!(x instanceof GpuFloatVector gpuX))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + x.getClass());

        if (!(result instanceof GpuFloatVector gpuResult))
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

        if (!(x instanceof GpuFloatVector gpuX))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + x.getClass());

        if (!(result instanceof GpuFloatVector gpuResult))
            throw new UnsupportedOperationException("Unable to execute operation with a vector of class " + result.getClass());

        if (x.size() != rows)
            throw new IllegalArgumentException("x.size() != A.rows()");

        result.resize(cols);
        ensureTransposeIndex(executor);

        // every output element is assigned by the kernel, so no clear() is needed
        CSR_MATVEC_TRANSPOSE.use(cols,
                dValues, dTransPtr, dTransEntry,
                gpuX.buffer(), gpuResult.buffer(), executor.scalar(nnzPerRow), executor.scalar(cols));
    }

    /**
     * (Re)builds the column -> entries index used by {@link #transposeApply}: for every column {@code c},
     * {@code transEntry[transPtr[c] .. transPtr[c+1])} lists the flat indices {@code e = row * nnzPerRow + i} of the
     * entries whose column is {@code c} (a counting sort over {@link #colIndex}). Padding entries are included: their
     * value is 0, so they contribute nothing. Entries with an out-of-range column are skipped.
     */
    private void ensureTransposeIndex(GpuExecutor executor) {
        if (!transposeDirty && dTransPtr != null && dTransEntry != null)
            return;
        releaseTransposeIndex();

        int length = rows * nnzPerRow;
        int[] ptr = new int[cols + 1];
        for (int e = 0; e < length; e++) {
            int c = colIndex[e];
            if (c >= 0 && c < cols)
                ptr[c + 1]++;
        }
        for (int c = 0; c < cols; c++)
            ptr[c + 1] += ptr[c];

        int total = ptr[cols];
        int[] entry = new int[Math.max(total, 1)];
        int[] next = Arrays.copyOf(ptr, cols); // next free slot per column
        for (int e = 0; e < length; e++) {
            int c = colIndex[e];
            if (c >= 0 && c < cols)
                entry[next[c]++] = e;
        }

        dTransPtr = executor.allocateIntBuffer(cols + 1);
        dTransEntry = executor.allocateIntBuffer(entry.length);
        executor.uploadInts(dTransPtr, ptr, ptr.length);
        if (total > 0)
            executor.uploadInts(dTransEntry, entry, total);
        transposeDirty = false;
    }

    private void releaseTransposeIndex() {
        if (dTransPtr != null)
            dTransPtr.release();
        if (dTransEntry != null)
            dTransEntry.release();
        dTransPtr = null;
        dTransEntry = null;
        transposeDirty = true;
    }

    @Override
    public void add(int row, int col, double value) {
        int index = findIndex(row, col);

        values[index] = (float) (values[index] + value); // one rounding, from the (near-)exact sum

        requireExecutor().uploadFloatAt(dValues, index, values[index]);
    }

    @Override
    public void set(int row, int col, double value) {
        int index = findIndex(row, col);

        values[index] = (float) value;

        requireExecutor().uploadFloatAt(dValues, index, values[index]);
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
        float[] out = new float[rows.length];
        for (int i = 0; i < out.length; i++)
            out[i] = (float) get(rows[i], cols[i]);

        return new GpuFloatVector(requireExecutor(), out);
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
        boolean patternChanged = false;

        for (int i = 0; i < count; i++) {
            values[base + i] = (float) rowValues[i];
            if (colIndex[base + i] != rowCols[i])
                patternChanged = true;
            colIndex[base + i] = rowCols[i];
        }

        /*
         * Unused padded entries are zero.
         */
        for (int i = count; i < nnzPerRow; i++) {
            values[base + i] = 0.0f;
            int pad = row < cols ? row : 0;
            if (colIndex[base + i] != pad)
                patternChanged = true;
            colIndex[base + i] = pad;
        }
        if (patternChanged)
            transposeDirty = true;

        requireExecutor().uploadFloats(dValues, base, values, base, nnzPerRow);
        requireExecutor().uploadInts(dColIndex, base, colIndex, base, nnzPerRow);
    }

    @Override
    public double[] getRowValues(int r) {
        int base = r * nnzPerRow;

        double[] out = new double[nnzPerRow];
        for (int i = 0; i < nnzPerRow; i++)
            out[i] = values[base + i];

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
        if (newCols < 0)
            throw new IllegalArgumentException("newCols < 0");

        int newLength = newRows * nnzPerRow;

        values = Arrays.copyOf(values, newLength);
        colIndex = Arrays.copyOf(colIndex, newLength);

        rows = newRows;
        cols = newCols;

        GpuExecutor executor = requireExecutor();

        GpuResource newValues = executor.allocateFloatBuffer(Math.max(newLength, 1));
        GpuResource newColIndex = executor.allocateIntBuffer(Math.max(newLength, 1));

        if (newLength > 0) {
            executor.uploadFloats(newValues, values, newLength);
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
        releaseTransposeIndex(); // shape/pattern changed
    }

    @Override
    public void multiply(double[] x, double[] result) {
        throw new UnsupportedOperationException("Use GPU FloatVector operations instead");
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

    /** The host mirror of the values as a (widened) copy -- unlike {@link GpuPaddedCSRMatrixD#getValues()}, writes to it don't reach the matrix. */
    public double[] getValues() {
        double[] out = new double[values.length];
        for (int i = 0; i < out.length; i++)
            out[i] = values[i];
        return out;
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

        releaseTransposeIndex();
    }
}