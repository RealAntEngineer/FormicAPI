package com.rae.formicapi.foundation.math.operators.backend.gpu;

import org.jocl.NativePointerObject;

/**
 * A kernel/shader. Definition and bound-dispatch state live on the same
 * object -- no separate "unbound definition" vs "bound handle" type, and
 * no separate interface for reduction-shaped kernels: a reduction is just
 * a kernel whose result happens to be written to a small buffer passed as
 * one of {@code args}, like any other output. Reading that buffer back
 * afterwards is the caller's job (see {@code GpuDoubleVector.reduceSum}),
 * the same way a matrix's {@code apply()} already writes into a
 * {@code result} buffer the caller reads separately -- {@code use} is the
 * only dispatch primitive there is.
 *
 * <p>Binds once against one device (see {@link #bind}); after that,
 * {@link #use} dispatches directly against the cached native handle, with
 * no lookup of any kind involved.
 */
public interface Kernel {

    /**
     * Compiles (if needed) and binds this kernel against {@code context}'s
     * device/queue. Call once -- typically from
     * {@link GpuExecutable#bindKernels} when an executor is attached, not
     * on every dispatch. Returns {@code this} for chaining.
     */
    Kernel bind(GpuExecutor context);

    /**
     * Binds {@code args} in declaration order and dispatches over
     * {@code globalSize} work-items/threads. Does not block -- correctness
     * across ops on the same queue is already guaranteed by OpenCL's
     * default in-order execution, so this is fine for chaining further GPU
     * work.
     */
    void use(int globalSize, GpuResource... args);

    /**
     * Same as {@link #use}, but blocks the calling thread until this
     * specific dispatch has finished on the device before returning.
     *
     * <p>Use this when the caller is about to do something the queue's own
     * ordering doesn't cover -- release or reuse a buffer this dispatch
     * wrote to, hand it to another API, read a result on the host without
     * going through an already-blocking transfer, etc. Waits only on this
     * one command (via its own event), not the whole queue's backlog, so
     * it doesn't stall unrelated work queued by other vectors sharing the
     * same executor the way {@code executor.finish()} would.
     */
    void useBlocking(int globalSize, GpuResource... args);
}