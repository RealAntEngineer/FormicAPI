package com.rae.formicapi.foundation.math.operators.backend.gpu.opencl;

import com.rae.formicapi.foundation.math.operators.backend.gpu.GpuExecutor;
import com.rae.formicapi.foundation.math.operators.backend.gpu.GpuResource;
import com.rae.formicapi.foundation.math.operators.backend.gpu.ResourceEvent;
import org.jetbrains.annotations.Nullable;
import org.jocl.*;

import static org.jocl.CL.*;

/**
 * One OpenCL kernel argument -- everything a single {@code clSetKernelArg(...)}
 * call needs: a size in bytes and a {@link Pointer} ({@code null} for
 * {@code __local} scratch args, where OpenCL wants only a size and no host
 * pointer at all).
 */
public final class OpenCLResource implements GpuResource {

    private static final cl_event[] NONE = new cl_event[0];

    final long size;
    final cl_mem pointer;
    @Nullable cl_event[] events;

    private OpenCLResource(long size, cl_mem pointer, @Nullable cl_event[] events) {
        this.size = size;
        this.pointer = pointer;
        this.events = events;
    }

    public void waitForEventsAndClean() {
        clWaitForEvents(events.length, events);
        for (cl_event event : events){
            clReleaseEvent(event);
        }
        events = NONE;
    }

    @Override
    public void add(ResourceEvent newEvent) {
        if (newEvent instanceof OpenCLEvent(cl_event event)) {
            cl_event[] oldEvents = events;
            events = new cl_event[events.length + 1];
            System.arraycopy(oldEvents, 0, events, 0, oldEvents.length);
            events[oldEvents.length] = event;
        }
    }

    @Override
    public void release() {
        waitForEventsAndClean();
        clReleaseMemObject(pointer);
    }

    public static OpenCLResource buffer(cl_mem buffer) {
        return new OpenCLResource(Sizeof.cl_mem, buffer, NONE);
    }

    public static OpenCLResource of(GpuExecutor executor, int value) {
        return new OpenCLResource(Sizeof.cl_int, clCreateBuffer(((OpenCLGpuExecutor)executor).context(),
                CL_MEM_READ_WRITE, Sizeof.cl_int, Pointer.to(new int[]{value}), null), NONE);
    }

    public static OpenCLResource of(GpuExecutor executor, double value) {
        return new OpenCLResource(Sizeof.cl_double, clCreateBuffer(((OpenCLGpuExecutor)executor).context(),
                CL_MEM_READ_WRITE, Sizeof.cl_double, Pointer.to(new double[]{value}), null),NONE);
    }

    //TODO The cl_mem can't be null. Why do we need that ?
    /** {@code __local} scratch space, sized in bytes -- no host-side data. */
    public static OpenCLResource local(GpuExecutor executor, long bytes) {
        return new OpenCLResource(bytes, clCreateBuffer(((OpenCLGpuExecutor)executor).context(),
                CL_MEM_READ_WRITE, Sizeof.cl_int, Pointer.to(new byte[(int)bytes]), null), NONE);
    }
}