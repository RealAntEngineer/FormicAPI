package com.rae.formicapi.foundation.math.operators.backend.gpu.opencl;

import com.rae.formicapi.foundation.math.operators.backend.gpu.GpuExecutor;
import com.rae.formicapi.foundation.math.operators.backend.gpu.GpuResource;
import org.jocl.*;

import static org.jocl.CL.*;

/**
 * OpenCL's {@link GpuExecutor}: owns the platform/device/context/queue and
 * implements the full buffer lifecycle. Deliberately owns no kernel source
 * and no launch methods -- those live as {@link OpenCLKernel} constants
 * defined next to whatever {@code Gpu*Vector}/{@code Gpu*Matrix} class uses
 * them, compiled lazily via {@link OpenCLKernel#bind}. Adding a new GPU
 * operation means adding a new {@code Kernel} constant at its point of
 * use; this class never needs to change for it.
 *
 * <p>Methods beyond the {@link GpuExecutor} contract ({@link #device()},
 * {@link #context()}, {@link #queue()}, {@link #supports(OpenCLKernel)})
 * are OpenCL-specific and only ever called by {@link OpenCLKernel} itself,
 * which already knows it's OpenCL-only -- nothing in the
 * {@code Gpu*Vector}/{@code Gpu*Matrix} layer above should reference this
 * class by name at all, only {@link GpuExecutor}.
 *
 * <p><b>Thread-safety.</b> Buffer operations here are {@code synchronized}
 * for the same reason kernel dispatch is: a single {@code cl_command_queue}
 * is not designed for concurrent multithreaded dispatch from several Java
 * threads at once.
 *
 * <p><b>Precision.</b> Requires a device supporting {@code cl_khr_fp64}
 * (double precision). Selection fails fast with a clear message if none is
 * found, rather than silently truncating to float.
 */
public final class OpenCLGpuExecutor implements GpuExecutor, AutoCloseable {

    //TODO remove some of the useless blocks
    private final cl_platform_id   platform;
    private final cl_device_id     device;
    private final cl_context       context;
    private final cl_command_queue queue;

    private volatile boolean closed = false;

    /** Opens a context on the first fp64-capable GPU device found; falls back to any fp64-capable device. */
    public OpenCLGpuExecutor() {
        this(selectDefaultDevice());
    }

    public OpenCLGpuExecutor(cl_device_id device) {
        CL.setExceptionsEnabled(true);

        this.device = device;
        this.platform = queryPlatform(device);

        cl_context_properties props = new cl_context_properties();
        props.addProperty(CL_CONTEXT_PLATFORM, platform);

        this.context = clCreateContext(props, 1, new cl_device_id[]{device}, null, null, null);
        this.queue = clCreateCommandQueue(context, device, 0, null);
        // Kernels are no longer pre-compiled here -- each OpenCLKernel
        // compiles itself lazily, the first time it's bind()ed against
        // this executor. See OpenCLKernel's class doc.
    }

    // Package-private, OpenCL-specific: only OpenCLKernel needs these, and only it should ever reference this class instead of GpuContext.
    cl_device_id device() { return device; }
    cl_context context() { return context; }
    cl_command_queue queue() { return queue; }

    // ---------------------------------------------------------------- device selection

    private static cl_device_id selectDefaultDevice() {
        CL.setExceptionsEnabled(true);

        int[] numPlatforms = new int[1];
        clGetPlatformIDs(0, null, numPlatforms);
        if (numPlatforms[0] == 0)
            throw new RuntimeException("No OpenCL platforms found");

        cl_platform_id[] platforms = new cl_platform_id[numPlatforms[0]];
        clGetPlatformIDs(platforms.length, platforms, null);

        cl_device_id fallback = null;

        for (cl_platform_id platform : platforms) {
            for (cl_device_id candidate : devicesOf(platform, CL_DEVICE_TYPE_GPU)) {
                if (supportsFp64(candidate))
                    return candidate;
                System.out.println("GPU not supporting F64");
            }
            System.out.println("No gpu defaulting to CPU");
            for (cl_device_id candidate : devicesOf(platform, CL_DEVICE_TYPE_ALL)) {
                if (fallback == null && supportsFp64(candidate))
                    fallback = candidate;
            }
        }

        if (fallback == null)
            throw new RuntimeException("No OpenCL device with cl_khr_fp64 (double precision) support found");

        return fallback;
    }

    private static cl_platform_id queryPlatform(cl_device_id device) {
        cl_platform_id[] out = new cl_platform_id[1];
        clGetDeviceInfo(device, CL_DEVICE_PLATFORM, Sizeof.cl_platform_id, Pointer.to(out), null);
        return out[0];
    }

    private static cl_device_id[] devicesOf(cl_platform_id platform, long deviceType) {
        int[] numDevices = new int[1];
        try {
            clGetDeviceIDs(platform, deviceType, 0, null, numDevices);
        } catch (CLException e) {
            return new cl_device_id[0]; // e.g. CL_DEVICE_NOT_FOUND when a platform has no GPU
        }
        if (numDevices[0] == 0)
            return new cl_device_id[0];

        cl_device_id[] devices = new cl_device_id[numDevices[0]];
        clGetDeviceIDs(platform, deviceType, devices.length, devices, null);
        return devices;
    }

    private static boolean supportsFp64(cl_device_id device) {
        return deviceExtensions(device).contains("cl_khr_fp64");
    }

    private static String deviceExtensions(cl_device_id device) {
        long[] size = new long[1];
        clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, null, size);
        byte[] buffer = new byte[(int) size[0]];
        clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, buffer.length, Pointer.to(buffer), null);
        return new String(buffer).trim();
    }

    /** True if this device supports a given kernel's required extensions -- check before relying on an optional kernel (e.g. atomic scatter-add). OpenCL-specific, see class doc. */
    public boolean supports(OpenCLKernel kernel) {
        return kernel.isSupported(device);
    }

    @Override
    public long maxWorkGroupSize() {
        long[] out = new long[1];
        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, Sizeof.size_t, Pointer.to(out), null);
        return out[0];
    }

    // ---------------------------------------------------------------- buffer lifecycle

    @Override
    public GpuResource allocateDoubleBuffer(int length) {
        return OpenCLResource.buffer(clCreateBuffer(context, CL_MEM_READ_WRITE, (long) length * Sizeof.cl_double,
                null, null));
    }

    @Override
    public GpuResource allocateIntBuffer(int length) {
        return OpenCLResource.buffer(clCreateBuffer(context, CL_MEM_READ_WRITE, (long) length * Sizeof.cl_int,
                null, null));
    }

    @Override
    public GpuResource allocateByteBuffer(int length) {

        return OpenCLResource.buffer(clCreateBuffer(context, CL_MEM_READ_WRITE, (long) length * Sizeof.cl_char,
                null, null));
    }

    @Override
    public synchronized void uploadDoubles(GpuResource buffer, double[] host, int length) {
        uploadDoubles(buffer, 0, host, 0, length);
    }

    @Override
    public synchronized void uploadDoubles(GpuResource buffer, int elementOffset, double[] host, int hostOffset, int length) {
        buffer.waitForEventsAndClean();
        cl_event event = new cl_event();
        buffer.add(new OpenCLEvent(event));
        clEnqueueWriteBuffer(queue, mem(buffer), CL_TRUE, (long) elementOffset * Sizeof.cl_double,
                (long) length * Sizeof.cl_double, Pointer.to(host).withByteOffset((long) hostOffset * Sizeof.cl_double),
                0, null, event);
    }

    @Override
    public synchronized void downloadDoubles(GpuResource buffer, double[] host, int length) {
        downloadDoubles(buffer, 0, host, length);
    }

    @Override
    public synchronized void downloadDoubles(GpuResource buffer, int elementOffset, double[] host, int length) {
        buffer.waitForEventsAndClean();
        cl_event event = new cl_event();
        buffer.add(new OpenCLEvent(event));
        clEnqueueReadBuffer(queue, mem(buffer), CL_TRUE, (long) elementOffset * Sizeof.cl_double,
                (long) length * Sizeof.cl_double, Pointer.to(host), 0, null, event);
    }

    @Override
    public synchronized void uploadInts(GpuResource buffer, int[] host, int length) {
        uploadInts(buffer, 0, host, 0, length);
    }

    @Override
    public synchronized void uploadInts(GpuResource buffer, int elementOffset, int[] host, int hostOffset, int length) {
        buffer.waitForEventsAndClean();
        cl_event event = new cl_event();
        buffer.add(new OpenCLEvent(event));
        clEnqueueWriteBuffer(queue, mem(buffer), CL_TRUE, (long) elementOffset * Sizeof.cl_int,
                (long) length * Sizeof.cl_int, Pointer.to(host).withByteOffset((long) hostOffset * Sizeof.cl_int),
                0, null, event);
    }

    @Override
    public synchronized void downloadInts(GpuResource buffer, int[] host, int length) {
        downloadInts(buffer, 0, host, length);
    }

    @Override
    public synchronized void downloadInts(GpuResource buffer, int elementOffset, int[] host, int length) {
        buffer.waitForEventsAndClean();
        cl_event event = new cl_event();
        buffer.add(new OpenCLEvent(event));
        clEnqueueReadBuffer(queue, mem(buffer), CL_TRUE, (long) elementOffset * Sizeof.cl_int,
                (long) length * Sizeof.cl_int, Pointer.to(host), 0, null, event);
    }

    @Override
    public synchronized void uploadDoubleAt(GpuResource buffer, int elementOffset, double value) {
        buffer.waitForEventsAndClean();
        cl_event event = new cl_event();
        buffer.add(new OpenCLEvent(event));
        clEnqueueWriteBuffer(queue, mem(buffer), CL_TRUE, (long) elementOffset * Sizeof.cl_double,
                Sizeof.cl_double, Pointer.to(new double[]{value}), 0, null, event);
    }

    @Override
    public synchronized double downloadDoubleAt(GpuResource buffer, int elementOffset) {
        double[] out = new double[1];
        buffer.waitForEventsAndClean();
        cl_event event = new cl_event();
        buffer.add(new OpenCLEvent(event));
        clEnqueueReadBuffer(queue, mem(buffer), CL_TRUE, (long) elementOffset * Sizeof.cl_double,
                Sizeof.cl_double, Pointer.to(out), 0, null, event);
        return out[0];
    }

    @Override
    public synchronized void uploadIntAt(GpuResource buffer, int elementOffset, int value) {
        buffer.waitForEventsAndClean();
        cl_event event = new cl_event();
        buffer.add(new OpenCLEvent(event));
        clEnqueueWriteBuffer(queue, mem(buffer), CL_TRUE, (long) elementOffset * Sizeof.cl_int,
                Sizeof.cl_int, Pointer.to(new int[]{value}), 0, null, event);
    }

    @Override
    public synchronized int downloadIntAt(GpuResource buffer, int elementOffset) {
        int[] out = new int[1];
        buffer.waitForEventsAndClean();
        cl_event event = new cl_event();
        buffer.add(new OpenCLEvent(event));
        clEnqueueReadBuffer(queue, mem(buffer), CL_TRUE, (long) elementOffset * Sizeof.cl_int,
                Sizeof.cl_int, Pointer.to(out), 0, null, event);
        return out[0];
    }

    @Override
    public synchronized void uploadBytes(GpuResource buffer, byte[] host, int length) {
        buffer.waitForEventsAndClean();
        cl_event event = new cl_event();
        buffer.add(new OpenCLEvent(event));
        clEnqueueWriteBuffer(queue, mem(buffer), CL_TRUE, 0, length, Pointer.to(host),
                0, null, event);
    }

    @Override
    public synchronized void downloadBytes(GpuResource buffer, byte[] host, int length) {
        buffer.waitForEventsAndClean();
        cl_event event = new cl_event();
        buffer.add(new OpenCLEvent(event));
        clEnqueueReadBuffer(queue, mem(buffer), CL_TRUE, 0, length, Pointer.to(host),
                0, null, event);
    }

    @Override
    public synchronized void uploadByteAt(GpuResource buffer, int elementOffset, byte value) {
        buffer.waitForEventsAndClean();
        cl_event event = new cl_event();
        buffer.add(new OpenCLEvent(event));
        clEnqueueWriteBuffer(queue, mem(buffer), CL_TRUE, elementOffset, 1, Pointer.to(new byte[]{value}),
                0, null, event);
    }

    @Override
    public synchronized byte downloadByteAt(GpuResource buffer, int elementOffset) {
        byte[] out = new byte[1];
        buffer.waitForEventsAndClean();
        cl_event event = new cl_event();
        buffer.add(new OpenCLEvent(event));
        clEnqueueReadBuffer(queue, mem(buffer), CL_TRUE, elementOffset, 1, Pointer.to(out),
                0, null, event);
        return out[0];
    }

    @Override
    public synchronized void copyDoubleBuffer(GpuResource src, GpuResource dst, int length) {
        if (length <= 0) return;
        src.waitForEventsAndClean();
        dst.waitForEventsAndClean();
        cl_event event = new cl_event();
        src.add(new OpenCLEvent(event));
        dst.add(new OpenCLEvent(event));
        clEnqueueCopyBuffer(queue, mem(src), mem(dst), 0, 0, (long) length * Sizeof.cl_double,
                0, null, event);
        //clFinish(queue);
    }

    @Override
    public synchronized void copyIntBuffer(GpuResource src, GpuResource dst, int length) {
        if (length <= 0) return;
        src.waitForEventsAndClean();
        dst.waitForEventsAndClean();
        cl_event event = new cl_event();
        src.add(new OpenCLEvent(event));
        dst.add(new OpenCLEvent(event));
        clEnqueueCopyBuffer(queue, mem(src), mem(dst), 0, 0, (long) length * Sizeof.cl_int,
                0, null, event);
        //clFinish(queue);
    }

    @Override
    public synchronized void copyByteBuffer(GpuResource src, GpuResource dst, int length) {
        if (length <= 0) return;        src.waitForEventsAndClean();
        dst.waitForEventsAndClean();
        cl_event event = new cl_event();
        src.add(new OpenCLEvent(event));
        dst.add(new OpenCLEvent(event));
        clEnqueueCopyBuffer(queue, mem(src), mem(dst), 0, 0, length,
                0, null, null);
        //clFinish(queue);
    }

    @Override
    public synchronized void fillDoubleBuffer(GpuResource buffer, double value, int length) {
        if (length <= 0) return;
        buffer.waitForEventsAndClean();
        cl_event event = new cl_event();
        buffer.add(new OpenCLEvent(event));
        clEnqueueFillBuffer(queue, mem(buffer), Pointer.to(new double[]{value}), Sizeof.cl_double,
                0,(long) length * Sizeof.cl_double, 0, null, null);
    }

    @Override
    public synchronized void fillIntBuffer(GpuResource buffer, int value, int length) {
        if (length <= 0) return;
        buffer.waitForEventsAndClean();
        cl_event event = new cl_event();
        buffer.add(new OpenCLEvent(event));
        clEnqueueFillBuffer(queue, mem(buffer), Pointer.to(new int[]{value}), Sizeof.cl_int,
                0,(long) length * Sizeof.cl_int, 0, null, null);
    }

    @Override
    public synchronized void fillByteBuffer(GpuResource buffer, byte value, int length) {
        if (length <= 0) return;
        buffer.waitForEventsAndClean();
        cl_event event = new cl_event();
        buffer.add(new OpenCLEvent(event));
        clEnqueueFillBuffer(queue, mem(buffer), Pointer.to(new byte[]{value}), Sizeof.cl_char,
                0, length, 0, null, null);
    }

    private static cl_mem mem(GpuResource buffer) {
        return ((OpenCLResource) buffer).pointer;
    }


    @Override
    @Deprecated//rely on the resource aware system instead.
    public void finish() {
        clFinish(queue);
    }

    @Override
    public synchronized void close() {
        if (closed) return;
        closed = true;
        OpenCLKernel.releaseAll(queue); // only kernels currently bound to this executor's queue
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
    }
}