package com.rae.formicapi.foundation.math.operators.backend.gpu.opencl;

import com.rae.formicapi.foundation.math.operators.backend.gpu.GpuResource;
import com.rae.formicapi.foundation.math.operators.backend.gpu.ResourceEvent;
import org.jetbrains.annotations.Nullable;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_event;
import org.jocl.cl_mem;

import static org.jocl.CL.clReleaseEvent;
import static org.jocl.CL.clReleaseMemObject;

/**
 * One OpenCL kernel argument / device resource.
 *
 * <p>Two different things share this type, deliberately: an actual device
 * buffer ({@link #buffer}, backed by a real {@code cl_mem}, with a
 * lifetime and outstanding-event tracking other operations depend on) and
 * an immediate scalar or {@code __local} scratch argument
 * ({@link #of}/{@link #local}, which is just bytes handed to
 * {@code clSetKernelArg} for one call -- no {@code cl_mem}, nothing to
 * track, nothing to release). Unifying them under one type is what lets
 * {@code Kernel#use} take a single flat argument list; {@link #kind} is
 * what keeps that unification from corrupting scalar arguments into bogus
 * memory-object pointers -- a kernel parameter declared {@code const
 * double a} wants the actual bytes of {@code a} passed directly, not a
 * pointer to some unrelated buffer that happens to hold it, and {@code
 * __local} parameters take no pointer at all (OpenCL allocates that
 * scratch space per work-group itself; {@code clSetKernelArg} just wants a
 * size and a null pointer for those).
 */
public final class OpenCLResource implements GpuResource {

    private static final cl_event[] NONE = new cl_event[0];

    private enum Kind { BUFFER, IMMEDIATE, LOCAL }

    final long size;
    final @Nullable Pointer pointer; // set for BUFFER (points at mem) and IMMEDIATE (points at the value); null for LOCAL
    private final Kind kind;
    private final @Nullable cl_mem mem; // set only for BUFFER

    private cl_event[] events = NONE;

    private OpenCLResource(long size, Kind kind, @Nullable cl_mem mem, @Nullable Pointer pointer) {
        this.size = size;
        this.kind = kind;
        this.mem = mem;
        this.pointer = pointer;
    }

    public static OpenCLResource buffer(cl_mem mem) {
        return new OpenCLResource(Sizeof.cl_mem, Kind.BUFFER, mem, Pointer.to(mem));
    }

    public static OpenCLResource of(int value) {
        return new OpenCLResource(Sizeof.cl_int, Kind.IMMEDIATE, null, Pointer.to(new int[]{value}));
    }

    public static OpenCLResource of(double value) {
        return new OpenCLResource(Sizeof.cl_double, Kind.IMMEDIATE, null, Pointer.to(new double[]{value}));
    }

    /** {@code __local} scratch space, sized in bytes -- no {@code cl_mem}; OpenCL allocates this per work-group at launch, not us. */
    public static OpenCLResource local(long bytes) {
        return new OpenCLResource(bytes, Kind.LOCAL, null, null);
    }

    boolean isBuffer() {
        return kind == Kind.BUFFER;
    }

    cl_mem mem() {
        if (mem == null)
            throw new IllegalStateException("Not a buffer resource (kind=" + kind + ")");
        return mem;
    }

    /** Package-private raw accessor, avoids wrapping into ResourceEvent[] when both sides are already OpenCL-internal (OpenCLKernel, OpenCLGpuExecutor). */
    synchronized cl_event[] rawEvents() {
        return events;
    }

    @Override
    public synchronized ResourceEvent[] pendingEvents() {
        ResourceEvent[] out = new OpenCLEvent[events.length];
        for (int i = 0; i < events.length; i++)
            out[i] = new OpenCLEvent(events[i]);
        return out;
    }

    @Override
    public synchronized void recordEvent(ResourceEvent newEvent) {
        if (kind != Kind.BUFFER) return; // nothing to depend on for an immediate/local arg
        if (!(newEvent instanceof OpenCLEvent(cl_event event)))
            throw new IllegalArgumentException("Not an OpenCL event: " + newEvent.getClass());
        releaseAll(events);
        events = new cl_event[]{event};
    }

    @Override
    public synchronized void settle() {
        if (kind != Kind.BUFFER) return;//WAIT FIRST !!!!!
        releaseAll(events);
        events = NONE;
    }

    private static void releaseAll(cl_event[] events) {
        for (cl_event e : events)
            clReleaseEvent(e);
    }

    @Override
    public void release() {
        if (kind != Kind.BUFFER) return; // nothing was ever allocated for IMMEDIATE/LOCAL
        settle();
        clReleaseMemObject(mem());
    }
}