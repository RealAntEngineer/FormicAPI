package com.rae.formicapi.math_tests.backend.gpu;

import com.rae.formicapi.foundation.math.operators.backend.gpu.*;
import com.rae.formicapi.foundation.math.operators.linear.MutableMatrix;
import com.rae.formicapi.foundation.math.operators.vectors.RealVector;
import org.junit.jupiter.api.DynamicTest;

import java.util.Arrays;
import java.util.function.Consumer;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * The two floating-point flavours of every GPU class ({@code GpuDoubleVector}/{@code GpuFloatVector},
 * {@code GpuPaddedCSRMatrixD}/{@code F}, ...) and everything that differs between them, in one place, so a
 * contract is written once and runs for both instead of being copied.
 *
 * <p>All test data is passed around as {@code double[]}. For {@link #FLOAT} it must already be rounded with
 * {@link #round}: the reference implementations are then fed the very same values the GPU stores, so a failure
 * means the GPU arithmetic is wrong, not that float32 is less precise than float64.
 *
 * <p>{@link #FLOAT} runs on every device. {@link #DOUBLE} needs fp64; callers gate on {@link #available}, which makes
 * the double flavour <em>skip</em> (not fail) on devices without it.
 */
public enum GpuTestPrecision {
    DOUBLE(1e-12, 1e-10),
    FLOAT(2e-4, 1e-4);

    /** Relative tolerance for elementwise results (and short chains of them), measured against max(1, |expected|). */
    public final double elementTolerance;
    /** Relative tolerance for reductions such as dot products over up to 100k elements. */
    public final double reductionTolerance;

    GpuTestPrecision(double elementTolerance, double reductionTolerance) {
        this.elementTolerance = elementTolerance;
        this.reductionTolerance = reductionTolerance;
    }

    public boolean isFloat() {
        return this == FLOAT;
    }

    /**
     * One dynamic test per precision for {@code body}, for use from a {@code @TestFactory}: this is how a contract written
     * once runs for both flavours. Double is <em>skipped</em> (not failed) where {@code ex} has no fp64; float always runs.
     */
    public static Stream<DynamicTest> forEach(GpuExecutor ex, Consumer<GpuTestPrecision> body) {
        return Arrays.stream(values()).map(p -> DynamicTest.dynamicTest(p.name(), () -> {
            assumeTrue(p.available(ex), "device has no fp64, so the double classes can't run here");
            body.accept(p);
        }));
    }

    /** Whether this flavour can run on {@code ex}: float always, double only on devices with fp64. */
    public boolean available(GpuExecutor ex) {
        return isFloat() || GpuTestBackends.supportsDouble(ex);
    }

    /** {@code v} as this precision stores it. */
    public double round(double v) {
        return isFloat() ? (double) (float) v : v;
    }

    public double[] round(double[] a) {
        double[] out = new double[a.length];
        for (int i = 0; i < a.length; i++)
            out[i] = round(a[i]);
        return out;
    }

    /** Absolute tolerance for comparing one element against {@code expected}. */
    public double tolerance(double expected) {
        return elementTolerance * Math.max(1.0, Math.abs(expected));
    }

    // ------------------------------------------------------------------ GPU objects of this precision

    /** A GPU vector holding {@code data} (which must already be rounded to this precision). */
    public RealVector vector(GpuExecutor ex, double[] data) {
        return isFloat() ? new GpuFloatVector(ex, toFloat(data)) : new GpuDoubleVector(ex, data.clone());
    }

    public RealVector vector(GpuExecutor ex, int n) {
        return isFloat() ? new GpuFloatVector(ex, n) : new GpuDoubleVector(ex, n);
    }

    /** The contents of a GPU vector of either precision, widened to double. */
    public static double[] read(RealVector v) {
        if (v instanceof GpuFloatVector f)
            return widen(f.download());
        return ((GpuDoubleVector) v).download();
    }

    /** A padded-CSR matrix of this precision, already attached to {@code ex}. Release it with {@link #close(MutableMatrix)}. */
    public MutableMatrix matrix(GpuExecutor ex, int rows, int cols, int nnzPerRow) {
        if (isFloat()) {
            GpuPaddedCSRMatrixF m = new GpuPaddedCSRMatrixF(rows, cols, nnzPerRow);
            m.setExecutor(ex);
            return m;
        }
        GpuPaddedCSRMatrixD m = new GpuPaddedCSRMatrixD(rows, cols, nnzPerRow);
        m.setExecutor(ex);
        return m;
    }

    /** {@code close()} is not on {@link MutableMatrix}, so releasing a matrix of either precision goes through here. */
    public static void close(MutableMatrix m) {
        if (m instanceof GpuPaddedCSRMatrixF f) f.close();
        else if (m instanceof GpuPaddedCSRMatrixD d) d.close();
    }

    // ------------------------------------------------------------------ raw buffers (executor level)

    public GpuResource alloc(GpuExecutor ex, int n) {
        return isFloat() ? ex.allocateFloatBuffer(n) : ex.allocateDoubleBuffer(n);
    }

    public void upload(GpuExecutor ex, GpuResource b, double[] data) {
        upload(ex, b, 0, data, 0, data.length);
    }

    public void upload(GpuExecutor ex, GpuResource b, int elementOffset, double[] data, int hostOffset, int length) {
        if (isFloat())
            ex.uploadFloats(b, elementOffset, toFloat(data), hostOffset, length);
        else
            ex.uploadDoubles(b, elementOffset, data, hostOffset, length);
    }

    public double[] download(GpuExecutor ex, GpuResource b, int n) {
        return download(ex, b, 0, n);
    }

    public double[] download(GpuExecutor ex, GpuResource b, int elementOffset, int n) {
        if (isFloat()) {
            float[] f = new float[n];
            ex.downloadFloats(b, elementOffset, f, n);
            return widen(f);
        }
        double[] d = new double[n];
        ex.downloadDoubles(b, elementOffset, d, n);
        return d;
    }

    public void uploadAt(GpuExecutor ex, GpuResource b, int index, double v) {
        if (isFloat()) ex.uploadFloatAt(b, index, (float) v);
        else ex.uploadDoubleAt(b, index, v);
    }

    public double downloadAt(GpuExecutor ex, GpuResource b, int index) {
        return isFloat() ? ex.downloadFloatAt(b, index) : ex.downloadDoubleAt(b, index);
    }

    public void fill(GpuExecutor ex, GpuResource b, int offset, double v, int length) {
        if (isFloat()) ex.fillFloatBuffer(b, offset, (float) v, length);
        else ex.fillDoubleBuffer(b, offset, v, length);
    }

    public void copy(GpuExecutor ex, GpuResource src, int srcOffset, GpuResource dst, int dstOffset, int length) {
        if (isFloat()) ex.copyFloatBuffer(src, srcOffset, dst, dstOffset, length);
        else ex.copyDoubleBuffer(src, srcOffset, dst, dstOffset, length);
    }

    public void copyWithin(GpuExecutor ex, GpuResource b, int srcStart, int dstStart, int count) {
        if (isFloat()) ex.copyFloatWithin(b, srcStart, dstStart, count);
        else ex.copyDoubleWithin(b, srcStart, dstStart, count);
    }

    /** A kernel scalar argument of this precision. */
    public GpuResource scalar(GpuExecutor ex, double v) {
        return isFloat() ? ex.scalar((float) v) : ex.scalar(v);
    }

    public static float[] toFloat(double[] a) {
        float[] out = new float[a.length];
        for (int i = 0; i < a.length; i++)
            out[i] = (float) a[i];
        return out;
    }

    public static double[] widen(float[] a) {
        double[] out = new double[a.length];
        for (int i = 0; i < a.length; i++)
            out[i] = a[i];
        return out;
    }
}