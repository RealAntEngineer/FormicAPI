package com.rae.formicapi.foundation.math.operators.backend.gpu.opencl;

import com.rae.formicapi.foundation.math.operators.backend.gpu.GpuExecutor;
import com.rae.formicapi.foundation.math.operators.backend.gpu.Kernel;
import com.rae.formicapi.foundation.math.operators.backend.gpu.GpuResource;
import org.jetbrains.annotations.Nullable;
import org.jocl.*;

import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

import static org.jocl.CL.*;

/**
 * One OpenCL kernel. Holds its own compiled {@code cl_kernel} and
 * {@code cl_command_queue} directly as instance fields -- set once by
 * {@link #bind}, read directly (no map, no lookup) by every {@link #use}
 * call afterwards.
 *
 * <p><b>One bound device at a time.</b> Since {@code id`}/{@code queue}
 * are plain mutable fields rather than a per-device map, a kernel that has
 * been {@code bind()}ed is only valid against the device it was last bound
 * to. This is fine for the common case -- one {@code GpuExecutor} open at
 * a time, every {@code Gpu*Vector} sharing its static kernel constants --
 * but rebinding the same kernel to a second, different executor while the
 * first is still in use would make both silently race over the same
 * {@code id}/{@code queue} fields. If you need two executors genuinely
 * live at once, this needs a per-context registry again.
 *
 * <p>Self-registers into {@link #kernels} purely so {@link OpenCLGpuExecutor#close}
 * can release whichever kernels happen to be bound to its queue -- that
 * registry is never consulted on the dispatch path, only at shutdown.
 */
public class OpenCLKernel implements Kernel {

    private static final Set<OpenCLKernel> kernels = ConcurrentHashMap.newKeySet();

    protected final String   name;
    protected final String   source;
    protected final String[] requirements;

    private @Nullable cl_kernel        id;
    private @Nullable cl_command_queue queue;

    /**
     * @param name         name of the {@code __kernel} function in {@code source}
     * @param source       uncompiled OpenCL C source, containing at least one
     *                     {@code __kernel} function named {@code name}
     * @param requirements OpenCL extensions this kernel's source requires (e.g. "cl_khr_fp64")
     */
    public OpenCLKernel(String name, String source, String... requirements) {
        this.name = name;
        this.source = source;
        this.requirements = requirements;
        kernels.add(this);
    }

    /** True if this kernel's required extensions are all present on {@code device}. */
    public boolean isSupported(cl_device_id device) {
        Set<String> supported = Set.of(deviceExtensions(device).split("\\s+"));
        for (String requirement : requirements)
            if (!supported.contains(requirement))
                return false;
        return true;
    }

    @Override
    public synchronized Kernel bind(GpuExecutor context) {
        OpenCLGpuExecutor executor = (OpenCLGpuExecutor) context;

        if (id == null) {
            if (!isSupported(executor.device()))
                throw new UnsupportedOperationException(
                        "Kernel '" + name + "' requires extension(s) [" + String.join(", ", requirements) +
                                "], which this device doesn't report support for");
            id = compile(executor.context(), executor.device());
        }
        this.queue = executor.queue();
        return this;
    }

    private cl_kernel compile(cl_context context, cl_device_id device) {
        cl_program program = clCreateProgramWithSource(context, 1, new String[]{source}, null, null);
        try {
            clBuildProgram(program, 1, new cl_device_id[]{device}, "", null, null);
            return clCreateKernel(program, name, null);
        } catch (CLException e) {
            throw new RuntimeException("OpenCL kernel '" + name + "' build failed: " + buildLog(program, device), e);
        } finally {
            clReleaseProgram(program);
        }
    }

    @Override
    public void use(int globalSize, GpuResource... args) {
        dispatch(globalSize, args);
    }

    @Override
    public void useBlocking(int globalSize, GpuResource... args) {
        dispatch(globalSize, args);
    }

    private void dispatch(int globalSize, GpuResource[] args) {
        if (globalSize <= 0) return;

        cl_kernel        kernelId = requireId();
        cl_command_queue q        = requireQueue();

        synchronized (q) {
            cl_event event = new cl_event();
            for (int i = 0; i < args.length; i++) {
                if (args[i] instanceof OpenCLResource clArg) {
                    clArg.waitForEventsAndClean();
                    clArg.add(new OpenCLEvent(event));//this is kinda ugly, but that works.
                    clSetKernelArg(kernelId, i, clArg.size, Pointer.to(clArg.pointer));
                } else if (args[i] == null) {
                    clReleaseEvent(event);
                    throw new NullPointerException("Kernel Arg was null at index : "+ i);
                } else {
                    args[i].waitForEventsAndClean();
                    clReleaseEvent(event);
                    throw new RuntimeException("Invalid Implementation of a Kernel Arg for OpenCL : "+ args[i].getClass());
                }
            }

            clEnqueueNDRangeKernel(q, kernelId, 1, null, new long[]{globalSize}, null, 0, null, event);

        }
    }

    private cl_kernel requireId() {
        if (id == null)
            throw new IllegalStateException("Kernel '" + name + "' was never bind()ed -- call bind(executor) before use(...)");
        return id;
    }

    private cl_command_queue requireQueue() {
        if (queue == null)
            throw new IllegalStateException("Kernel '" + name + "' was never bind()ed -- call bind(executor) before use(...)");
        return queue;
    }

    private static String deviceExtensions(cl_device_id device) {
        long[] size = new long[1];
        clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, null, size);
        byte[] buffer = new byte[(int) size[0]];
        clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, buffer.length, Pointer.to(buffer), null);
        return new String(buffer).trim();
    }

    private static String buildLog(cl_program program, cl_device_id device) {
        long[] logSize = new long[1];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, null, logSize);
        byte[] log = new byte[(int) logSize[0]];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log.length, Pointer.to(log), null);
        return new String(log);
    }

    /** Releases this kernel's compiled handle if it's currently bound to {@code queue}, so a later bind() against a different device recompiles cleanly. */
    private synchronized void releaseIfBoundTo(cl_command_queue queue) {
        if (this.queue == queue && id != null) {
            clReleaseKernel(id);
            id = null;
            this.queue = null;
        }
    }

    /** Called from {@link OpenCLGpuExecutor#close()}: releases every kernel currently bound to that executor's queue, untouched otherwise. */
    static void releaseAll(cl_command_queue queue) {
        for (OpenCLKernel kernel : kernels)
            kernel.releaseIfBoundTo(queue);
    }
}