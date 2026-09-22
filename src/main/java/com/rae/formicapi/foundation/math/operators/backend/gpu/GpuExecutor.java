package com.rae.formicapi.foundation.math.operators.backend.gpu;

import com.rae.formicapi.foundation.math.operators.backend.gpu.opencl.OpenCLGpuExecutor;
import com.rae.formicapi.foundation.math.operators.backend.gpu.opencl.OpenCLKernel;

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
 * written against this interface only, so none of it needs to change when
 * a Vulkan backend is added (see the package doc).
 */
public interface GpuExecutor {

    GpuResource allocateDoubleBuffer(int length);
    GpuResource allocateIntBuffer(int length);
    GpuResource allocateByteBuffer(int length);

    //void release(GpuResource buffer);

    void uploadDoubles(GpuResource buffer, double[] host, int length);
    void uploadDoubles(GpuResource buffer, int elementOffset, double[] host, int hostOffset, int length);
    void downloadDoubles(GpuResource buffer, double[] host, int length);
    void downloadDoubles(GpuResource buffer, int elementOffset, double[] host, int length);
    void uploadDoubleAt(GpuResource buffer, int elementOffset, double value);
    double downloadDoubleAt(GpuResource buffer, int elementOffset);

    void uploadInts(GpuResource buffer, int[] host, int length);
    void uploadInts(GpuResource buffer, int elementOffset, int[] host, int hostOffset, int length);
    void downloadInts(GpuResource buffer, int[] host, int length);
    void downloadInts(GpuResource buffer, int elementOffset, int[] host, int length);
    void uploadIntAt(GpuResource buffer, int elementOffset, int value);
    int downloadIntAt(GpuResource buffer, int elementOffset);

    void uploadBytes(GpuResource buffer, byte[] host, int length);
    void downloadBytes(GpuResource buffer, byte[] host, int length);
    void uploadByteAt(GpuResource buffer, int elementOffset, byte value);
    byte downloadByteAt(GpuResource buffer, int elementOffset);

    void copyDoubleBuffer(GpuResource src, GpuResource dst, int length);
    void copyIntBuffer(GpuResource src, GpuResource dst, int length);
    void copyByteBuffer(GpuResource src, GpuResource dst, int length);

    void fillDoubleBuffer(GpuResource buffer, double value, int length);
    void fillIntBuffer(GpuResource buffer, int value, int length);
    void fillByteBuffer(GpuResource buffer, byte value, int length);

    /** This context's actual max work-group size, for callers sizing local/reduction work-groups instead of guessing a constant. */
    long maxWorkGroupSize();

    /** Blocks until every previously enqueued operation on this context has completed. */
    void finish();
}