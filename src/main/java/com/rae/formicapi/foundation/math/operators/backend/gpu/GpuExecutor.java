package com.rae.formicapi.foundation.math.operators.backend.gpu;

import com.rae.formicapi.foundation.math.operators.backend.BackendExecutor;
import com.rae.formicapi.foundation.math.operators.backend.gpu.opencl.OpenCLGpuExecutor;
import com.rae.formicapi.foundation.math.operators.backend.gpu.opencl.OpenCLKernel;
import com.rae.formicapi.foundation.math.operators.backend.gpu.vulkan.VulkanGpuExecutor;

/**
 * Everything a {@code Gpu*Vector}/{@code Gpu*Matrix} needs from whatever
 * device context it's attached to: buffer allocation, transfer, and
 * lifecycle, plus enough device-capability info to size dispatches (e.g.
 * {@link #maxWorkGroupSize()} for reduction kernels).
 *
 * <p>Deliberately does NOT expose anything OpenCL-specific -- no
 * {@code cl_context}/{@code cl_device_id}/{@code cl_command_queue}. Those
 * stay on {@link OpenCLGpuExecutor} itself, visible only to
 * {@link OpenCLKernel}, which already knows it's an OpenCL-only class and
 * casts a {@code GpuContext} down to the concrete type accordingly (the
 * same way {@code OpenCLKernelArg} unwraps a {@link GpuResource} down to a
 * {@code cl_mem}). Everything above that layer -- every
 * {@code Gpu*Vector}, {@code Gpu*Matrix}, {@link GpuExecutable} -- is
 * written against this interface only, so the same classes run unchanged on
 * {@link OpenCLGpuExecutor} and {@link VulkanGpuExecutor} (see the package
 * doc). Kernel arguments are created through {@link #scalar} and
 * {@link #localMemory} for the same reason.
 */
public interface GpuExecutor extends BackendExecutor {

    GpuResource allocateDoubleBuffer(int length);
    GpuResource allocateFloatBuffer(int length);
    GpuResource allocateIntBuffer(int length);
    GpuResource allocateByteBuffer(int length);

    //void release(GpuResource buffer);

    void uploadDoubles(GpuResource buffer, double[] host, int length);//TODO could be made default
    void uploadDoubles(GpuResource buffer, int elementOffset, double[] host, int hostOffset, int length);
    void downloadDoubles(GpuResource buffer, double[] host, int length);//TODO could be made default
    void downloadDoubles(GpuResource buffer, int elementOffset, double[] host, int length);
    void uploadDoubleAt(GpuResource buffer, int elementOffset, double value);
    double downloadDoubleAt(GpuResource buffer, int elementOffset);

    void uploadFloats(GpuResource buffer, float[] host, int length);//TODO could be made default
    void uploadFloats(GpuResource buffer, int elementOffset, float[] host, int hostOffset, int length);
    void downloadFloats(GpuResource buffer, float[] host, int length);//TODO could be made default
    void downloadFloats(GpuResource buffer, int elementOffset, float[] host, int length);
    void uploadFloatAt(GpuResource buffer, int elementOffset, float value);
    float downloadFloatAt(GpuResource buffer, int elementOffset);


    void uploadInts(GpuResource buffer, int[] host, int length);//TODO could be made default
    void uploadInts(GpuResource buffer, int elementOffset, int[] host, int hostOffset, int length);
    void downloadInts(GpuResource buffer, int[] host, int length);//TODO could be made default
    void downloadInts(GpuResource buffer, int elementOffset, int[] host, int length);
    void uploadIntAt(GpuResource buffer, int elementOffset, int value);
    int downloadIntAt(GpuResource buffer, int elementOffset);

    void uploadBytes(GpuResource buffer, byte[] host, int length);
    void downloadBytes(GpuResource buffer, byte[] host, int length);
    void uploadByteAt(GpuResource buffer, int elementOffset, byte value);
    byte downloadByteAt(GpuResource buffer, int elementOffset);

    default void copyDoubleBuffer(GpuResource src, GpuResource dst, int length) { copyDoubleBuffer(src, 0, dst, 0, length); }
    default void copyFloatBuffer(GpuResource src, GpuResource dst, int length) { copyFloatBuffer(src, 0, dst, 0, length); }

    default void copyIntBuffer(GpuResource src, GpuResource dst, int length)    { copyIntBuffer(src, 0, dst, 0, length); }
    default void copyByteBuffer(GpuResource src, GpuResource dst, int length)   { copyByteBuffer(src, 0, dst, 0, length); }

    /** Copies {@code length} elements from {@code src[srcOffset..]} to {@code dst[dstOffset..]} (element offsets). {@code src} and {@code dst} may be the same buffer only if the two ranges don't overlap -- see {@link #copyDoubleWithin}. */
    void copyDoubleBuffer(GpuResource src, int srcOffset, GpuResource dst, int dstOffset, int length);

    /** Copies {@code length} elements from {@code src[srcOffset..]} to {@code dst[dstOffset..]} (element offsets). {@code src} and {@code dst} may be the same buffer only if the two ranges don't overlap -- see {@link #copyFloatWithin}. */
    void copyFloatBuffer(GpuResource src, int srcOffset, GpuResource dst, int dstOffset, int length);

    /** Copies {@code length} elements from {@code src[srcOffset..]} to {@code dst[dstOffset..]} (element offsets). {@code src} and {@code dst} may be the same buffer only if the two ranges don't overlap -- see {@link #copyIntWithin}. */
    void copyIntBuffer(GpuResource src, int srcOffset, GpuResource dst, int dstOffset, int length);

    /** Copies {@code length} elements from {@code src[srcOffset..]} to {@code dst[dstOffset..]} (element offsets). {@code src} and {@code dst} may be the same buffer only if the two ranges don't overlap -- see {@link #copyByteWithin}. */
    void copyByteBuffer(GpuResource src, int srcOffset, GpuResource dst, int dstOffset, int length);

    /** {@code System.arraycopy}-style move inside one buffer: overlapping ranges are handled through a temporary. */
    default void copyDoubleWithin(GpuResource buffer, int srcStart, int dstStart, int count) {
        if (count <= 0 || srcStart == dstStart) return;
        if (Math.abs((long) srcStart - dstStart) >= count) {
            copyDoubleBuffer(buffer, srcStart, buffer, dstStart, count);
            return;
        }
        GpuResource tmp = allocateDoubleBuffer(count);
        try {
            copyDoubleBuffer(buffer, srcStart, tmp, 0, count);
            copyDoubleBuffer(tmp, 0, buffer, dstStart, count);
        } finally {
            tmp.release();
        }
    }

    default void copyFloatWithin(GpuResource buffer, int srcStart, int dstStart, int count) {
        if (count <= 0 || srcStart == dstStart) return;
        if (Math.abs((long) srcStart - dstStart) >= count) {
            copyFloatBuffer(buffer, srcStart, buffer, dstStart, count);
            return;
        }
        GpuResource tmp = allocateFloatBuffer(count);
        try {
            copyFloatBuffer(buffer, srcStart, tmp, 0, count);
            copyFloatBuffer(tmp, 0, buffer, dstStart, count);
        } finally {
            tmp.release();
        }
    }


    default void copyIntWithin(GpuResource buffer, int srcStart, int dstStart, int count) {
        if (count <= 0 || srcStart == dstStart) return;
        if (Math.abs((long) srcStart - dstStart) >= count) {
            copyIntBuffer(buffer, srcStart, buffer, dstStart, count);
            return;
        }
        GpuResource tmp = allocateIntBuffer(count);
        try {
            copyIntBuffer(buffer, srcStart, tmp, 0, count);
            copyIntBuffer(tmp, 0, buffer, dstStart, count);
        } finally {
            tmp.release();
        }
    }

    default void copyByteWithin(GpuResource buffer, int srcStart, int dstStart, int count) {
        if (count <= 0 || srcStart == dstStart) return;
        if (Math.abs((long) srcStart - dstStart) >= count) {
            copyByteBuffer(buffer, srcStart, buffer, dstStart, count);
            return;
        }
        GpuResource tmp = allocateByteBuffer(count);
        try {
            copyByteBuffer(buffer, srcStart, tmp, 0, count);
            copyByteBuffer(tmp, 0, buffer, dstStart, count);
        } finally {
            tmp.release();
        }
    }

    default void fillDoubleBuffer(GpuResource buffer, double value, int length) { fillDoubleBuffer(buffer, 0, value, length); }

    default void fillFloatBuffer(GpuResource buffer, float value, int length) { fillFloatBuffer(buffer, 0, value, length); }

    default void fillIntBuffer(GpuResource buffer, int value, int length)       { fillIntBuffer(buffer, 0, value, length); }

    default void fillByteBuffer(GpuResource buffer, byte value, int length) { fillByteBuffer(buffer, 0, value, length); }


    /** Fills {@code length} elements starting at element {@code offset}. */
    void fillDoubleBuffer(GpuResource buffer, int offset, double value, int length);

    /** Fills {@code length} elements starting at element {@code offset}. */
    void fillFloatBuffer(GpuResource buffer, int offset, float value, int length);

    /** Fills {@code length} elements starting at element {@code offset}. */
    void fillIntBuffer(GpuResource buffer, int offset, int value, int length);

    /** Fills {@code length} bytes starting at byte {@code offset}. */
    void fillByteBuffer(GpuResource buffer, int offset, byte value, int length);

    // ---------------------------------------------------------------- kernel arguments / kernels

    /** An immediate {@code int} kernel argument. */
    GpuResource scalar(int value);

    /** An immediate {@code double} kernel argument. */
    GpuResource scalar(double value);

    /** An immediate {@code float} kernel argument. */
    GpuResource scalar(float value);

    /**
     * Work-group scratch space of {@code bytes} bytes as a kernel argument.
     * Backends where scratch is declared inside the shader itself (Vulkan)
     * ignore it, so always size it as {@code localSize * elementSize} and
     * declare the shader's shared array with the same extent.
     */
    GpuResource localMemory(long bytes);

    /**
     * Builds this backend's native kernel from {@code definition}'s matching
     * source. Called by {@link PortableKernel#bind}; not meant for direct use.
     *
     * @throws UnsupportedOperationException if the definition has no source
     *         for this backend, or the device lacks something it requires
     */
    Kernel createKernel(PortableKernel definition);

    boolean supports(Kernel kernel);

    /** This context's actual max work-group size, for callers sizing local/reduction work-groups instead of guessing a constant. */
    long maxWorkGroupSize();

    /** Blocks until every previously enqueued operation on this context has completed. */
    void finish();
}