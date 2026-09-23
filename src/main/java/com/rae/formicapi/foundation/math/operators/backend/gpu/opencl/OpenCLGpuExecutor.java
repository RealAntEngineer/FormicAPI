package com.rae.formicapi.foundation.math.operators.backend.gpu.opencl;

import com.rae.formicapi.foundation.math.operators.backend.gpu.GpuExecutor;
import com.rae.formicapi.foundation.math.operators.backend.gpu.GpuResource;
import com.rae.formicapi.foundation.math.operators.backend.gpu.Kernel;
import org.jocl.*;

import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.Set;

import static org.jocl.CL.*;

/**
 * OpenCL's {@link GpuExecutor}: owns the platform/device/context/queue and
 * implements the full buffer lifecycle. Deliberately owns no kernel source
 * and no launch methods -- those live as {@link OpenCLKernel} constants
 * defined next to whatever {@code Gpu*Vector}/{@code Gpu*Matrix} class uses
 * them.
 *
 * <p><b>Locking.</b> Every method here synchronizes on {@code queue}, the
 * same monitor {@link OpenCLKernel#use} synchronizes on -- both sides
 * touch the same buffers' event state, so they need to share one lock, not
 * two different ones (an executor-instance lock here vs. a queue lock
 * there would let a buffer transfer and a kernel dispatch race on the same
 * buffer's pending-events list).
 *
 * <p><b>Events, not blocking waits.</b> Every method that touches a buffer
 * builds a wait list from that buffer's currently pending events and
 * passes it to the OpenCL call itself (a device-side wait), instead of
 * calling a blocking host-side wait first -- that would serialize the host
 * thread on every single call regardless of whether the two operations
 * actually depend on each other. Methods that block by nature (the
 * {@code CL_TRUE} reads/writes) call {@link GpuResource#settle()}
 * afterwards, since by then they're genuinely done; methods that don't
 * block (fill, copy) get a real output event and call
 * {@link GpuResource#recordEvent}, staying pending like a kernel dispatch
 * would.
 *
 * <p><b>Precision.</b> Requires a device supporting {@code cl_khr_fp64}
 * (double precision). Selection fails fast with a clear message if none is
 * found, rather than silently truncating to float.
 */
public final class OpenCLGpuExecutor implements GpuExecutor, AutoCloseable {

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
    }

    // Package-private, OpenCL-specific: only OpenCLKernel should ever reference this class instead of GpuExecutor.
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

    /** True if this device supports a given kernel's required extensions. OpenCL-specific, see class doc. */
    @Override
    public boolean supports(Kernel kernel) {
        return kernel instanceof OpenCLKernel clKernel && clKernel.isSupported(device);
    }

    @Override
    public long maxWorkGroupSize() {
        long[] out = new long[1];
        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, Sizeof.size_t, Pointer.to(out), null);
        return out[0];
    }

    // ---------------------------------------------------------------- allocation

    @Override
    public GpuResource allocateDoubleBuffer(int length) {
        return OpenCLResource.buffer(clCreateBuffer(context, CL_MEM_READ_WRITE, (long) length * Sizeof.cl_double, null, null));
    }

    @Override
    public GpuResource allocateIntBuffer(int length) {
        return OpenCLResource.buffer(clCreateBuffer(context, CL_MEM_READ_WRITE, (long) length * Sizeof.cl_int, null, null));
    }

    @Override
    public GpuResource allocateByteBuffer(int length) {
        return OpenCLResource.buffer(clCreateBuffer(context, CL_MEM_READ_WRITE, (long) length * Sizeof.cl_char, null, null));
    }

    // ---------------------------------------------------------------- wait-list / event-recording helpers

    private static cl_event[] waitList(GpuResource... resources) {
        Set<cl_event> set = new LinkedHashSet<>();
        for (GpuResource r : resources)
            if (r instanceof OpenCLResource clr)
                Collections.addAll(set, clr.rawEvents());
        return set.isEmpty() ? null : set.toArray(new cl_event[0]);
    }

    private static int waitCount(cl_event[] list) {
        return list == null ? 0 : list.length;
    }

    /** For non-blocking ops (fill, copy): record {@code event} on every touched resource, each holding its own independent reference. */
    private static void record(cl_event event, GpuResource... resources) {
        for (GpuResource r : resources) {
            clRetainEvent(event);
            r.recordEvent(new OpenCLEvent(event));
        }
        clReleaseEvent(event); // drop the creation-time reference
    }

    /** For genuinely blocking ops (CL_TRUE reads/writes): mark every touched resource fully settled. */
    private static void settle(GpuResource... resources) {
        for (GpuResource r : resources)
            r.settle();
    }

    private static cl_mem mem(GpuResource buffer) {
        return ((OpenCLResource) buffer).mem();
    }

    // ---------------------------------------------------------------- double transfers (blocking)

    @Override
    public void uploadDoubles(GpuResource buffer, double[] host, int length) {
        uploadDoubles(buffer, 0, host, 0, length);
    }

    @Override
    public void uploadDoubles(GpuResource buffer, int elementOffset, double[] host, int hostOffset, int length) {
        synchronized (queue) {
            cl_event[] wait = waitList(buffer);
            clEnqueueWriteBuffer(queue, mem(buffer), CL_TRUE, (long) elementOffset * Sizeof.cl_double,
                    (long) length * Sizeof.cl_double, Pointer.to(host).withByteOffset((long) hostOffset * Sizeof.cl_double),
                    waitCount(wait), wait, null);
            settle(buffer);
        }
    }

    @Override
    public void downloadDoubles(GpuResource buffer, double[] host, int length) {
        downloadDoubles(buffer, 0, host, length);
    }

    @Override
    public void downloadDoubles(GpuResource buffer, int elementOffset, double[] host, int length) {
        synchronized (queue) {
            cl_event[] wait = waitList(buffer);
            clEnqueueReadBuffer(queue, mem(buffer), CL_TRUE, (long) elementOffset * Sizeof.cl_double,
                    (long) length * Sizeof.cl_double, Pointer.to(host), waitCount(wait), wait, null);
            settle(buffer);
        }
    }

    @Override
    public void uploadDoubleAt(GpuResource buffer, int elementOffset, double value) {
        synchronized (queue) {
            cl_event[] wait = waitList(buffer);
            clEnqueueWriteBuffer(queue, mem(buffer), CL_TRUE, (long) elementOffset * Sizeof.cl_double,
                    Sizeof.cl_double, Pointer.to(new double[]{value}), waitCount(wait), wait, null);
            settle(buffer);
        }
    }

    @Override
    public double downloadDoubleAt(GpuResource buffer, int elementOffset) {
        synchronized (queue) {
            double[] out = new double[1];
            cl_event[] wait = waitList(buffer);
            clEnqueueReadBuffer(queue, mem(buffer), CL_TRUE, (long) elementOffset * Sizeof.cl_double,
                    Sizeof.cl_double, Pointer.to(out), waitCount(wait), wait, null);
            settle(buffer);
            return out[0];
        }
    }

    // ---------------------------------------------------------------- int transfers (blocking)

    @Override
    public void uploadInts(GpuResource buffer, int[] host, int length) {
        uploadInts(buffer, 0, host, 0, length);
    }

    @Override
    public void uploadInts(GpuResource buffer, int elementOffset, int[] host, int hostOffset, int length) {
        synchronized (queue) {
            cl_event[] wait = waitList(buffer);
            clEnqueueWriteBuffer(queue, mem(buffer), CL_TRUE, (long) elementOffset * Sizeof.cl_int,
                    (long) length * Sizeof.cl_int, Pointer.to(host).withByteOffset((long) hostOffset * Sizeof.cl_int),
                    waitCount(wait), wait, null);
            settle(buffer);
        }
    }

    @Override
    public void downloadInts(GpuResource buffer, int[] host, int length) {
        downloadInts(buffer, 0, host, length);
    }

    @Override
    public void downloadInts(GpuResource buffer, int elementOffset, int[] host, int length) {
        synchronized (queue) {
            cl_event[] wait = waitList(buffer);
            clEnqueueReadBuffer(queue, mem(buffer), CL_TRUE, (long) elementOffset * Sizeof.cl_int,
                    (long) length * Sizeof.cl_int, Pointer.to(host), waitCount(wait), wait, null);
            settle(buffer);
        }
    }

    @Override
    public void uploadIntAt(GpuResource buffer, int elementOffset, int value) {
        synchronized (queue) {
            cl_event[] wait = waitList(buffer);
            clEnqueueWriteBuffer(queue, mem(buffer), CL_TRUE, (long) elementOffset * Sizeof.cl_int,
                    Sizeof.cl_int, Pointer.to(new int[]{value}), waitCount(wait), wait, null);
            settle(buffer);
        }
    }

    @Override
    public int downloadIntAt(GpuResource buffer, int elementOffset) {
        synchronized (queue) {
            int[] out = new int[1];
            cl_event[] wait = waitList(buffer);
            clEnqueueReadBuffer(queue, mem(buffer), CL_TRUE, (long) elementOffset * Sizeof.cl_int,
                    Sizeof.cl_int, Pointer.to(out), waitCount(wait), wait, null);
            settle(buffer);
            return out[0];
        }
    }

    // ---------------------------------------------------------------- byte transfers (blocking)

    @Override
    public void uploadBytes(GpuResource buffer, byte[] host, int length) {
        synchronized (queue) {
            cl_event[] wait = waitList(buffer);
            clEnqueueWriteBuffer(queue, mem(buffer), CL_TRUE, 0, length, Pointer.to(host), waitCount(wait), wait, null);
            settle(buffer);
        }
    }

    @Override
    public void downloadBytes(GpuResource buffer, byte[] host, int length) {
        synchronized (queue) {
            cl_event[] wait = waitList(buffer);
            clEnqueueReadBuffer(queue, mem(buffer), CL_TRUE, 0, length, Pointer.to(host), waitCount(wait), wait, null);
            settle(buffer);
        }
    }

    @Override
    public void uploadByteAt(GpuResource buffer, int elementOffset, byte value) {
        synchronized (queue) {
            cl_event[] wait = waitList(buffer);
            clEnqueueWriteBuffer(queue, mem(buffer), CL_TRUE, elementOffset, 1, Pointer.to(new byte[]{value}), waitCount(wait), wait, null);
            settle(buffer);
        }
    }

    @Override
    public byte downloadByteAt(GpuResource buffer, int elementOffset) {
        synchronized (queue) {
            byte[] out = new byte[1];
            cl_event[] wait = waitList(buffer);
            clEnqueueReadBuffer(queue, mem(buffer), CL_TRUE, elementOffset, 1, Pointer.to(out), waitCount(wait), wait, null);
            settle(buffer);
            return out[0];
        }
    }

    // ---------------------------------------------------------------- copy / fill (non-blocking)

    @Override
    public void copyDoubleBuffer(GpuResource src, GpuResource dst, int length) {
        if (length <= 0) return;
        synchronized (queue) {
            cl_event[] wait = waitList(src, dst);
            cl_event event = new cl_event();
            clEnqueueCopyBuffer(queue, mem(src), mem(dst), 0, 0, (long) length * Sizeof.cl_double, waitCount(wait), wait, event);
            record(event, src, dst);
        }
    }

    @Override
    public void copyIntBuffer(GpuResource src, GpuResource dst, int length) {
        if (length <= 0) return;
        synchronized (queue) {
            cl_event[] wait = waitList(src, dst);
            cl_event event = new cl_event();
            clEnqueueCopyBuffer(queue, mem(src), mem(dst), 0, 0, (long) length * Sizeof.cl_int, waitCount(wait), wait, event);
            record(event, src, dst);
        }
    }

    @Override
    public void copyByteBuffer(GpuResource src, GpuResource dst, int length) {
        if (length <= 0) return;
        synchronized (queue) {
            cl_event[] wait = waitList(src, dst);
            cl_event event = new cl_event();
            clEnqueueCopyBuffer(queue, mem(src), mem(dst), 0, 0, length, waitCount(wait), wait, event);
            record(event, src, dst);
        }
    }

    @Override
    public void fillDoubleBuffer(GpuResource buffer, double value, int length) {
        if (length <= 0) return;
        synchronized (queue) {
            cl_event[] wait = waitList(buffer);
            cl_event event = new cl_event();
            clEnqueueFillBuffer(queue, mem(buffer), Pointer.to(new double[]{value}), Sizeof.cl_double,
                    0, (long) length * Sizeof.cl_double, waitCount(wait), wait, event);
            record(event, buffer);
        }
    }

    @Override
    public void fillIntBuffer(GpuResource buffer, int value, int length) {
        if (length <= 0) return;
        synchronized (queue) {
            cl_event[] wait = waitList(buffer);
            cl_event event = new cl_event();
            clEnqueueFillBuffer(queue, mem(buffer), Pointer.to(new int[]{value}), Sizeof.cl_int,
                    0, (long) length * Sizeof.cl_int, waitCount(wait), wait, event);
            record(event, buffer);
        }
    }

    @Override
    public void fillByteBuffer(GpuResource buffer, byte value, int length) {
        if (length <= 0) return;
        synchronized (queue) {
            cl_event[] wait = waitList(buffer);
            cl_event event = new cl_event();
            clEnqueueFillBuffer(queue, mem(buffer), Pointer.to(new byte[]{value}), Sizeof.cl_char,
                    0, length, waitCount(wait), wait, event);
            record(event, buffer);
        }
    }

    /** Full-queue barrier -- prefer letting the event graph handle dependencies; only reach for this for genuine external interop / debugging. */
    @Override
    public void finish() {
        clFinish(queue);
    }

    @Override
    public synchronized void close() {
        if (closed) return;
        closed = true;
        OpenCLKernel.releaseAll(queue);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
    }
}