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
 *
 * <p><b>Arguments.</b> {@code args} are positional and backend-neutral:
 * device buffers (from {@link GpuExecutor}'s {@code allocate*}), scalars
 * ({@link GpuExecutor#scalar(int)}/{@link GpuExecutor#scalar(double)}) and
 * work-group scratch ({@link GpuExecutor#localMemory(long)}). Never build
 * those through a concrete backend class -- that is what keeps every
 * {@code Gpu*Vector}/{@code Gpu*Matrix} usable on every backend.
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
     * {@code globalSize} work-items/threads, letting the driver pick the
     * work-group size. Does not block -- ordering across ops on the same
     * executor is guaranteed by the backend, so this is fine for chaining
     * further GPU work.
     */
    default void use(int globalSize, GpuResource... args) {
        use(globalSize, 0, args);
    }

    /**
     * Same as {@link #use(int, GpuResource...)}, but blocks the calling
     * thread until this specific dispatch has finished on the device before
     * returning.
     *
     * <p>Use this when the caller is about to do something the executor's
     * own ordering doesn't cover -- release or reuse a buffer this dispatch
     * wrote to, hand it to another API, read a result on the host without
     * going through an already-blocking transfer, etc.
     */
    default void useBlocking(int globalSize, GpuResource... args) {
        useBlocking(globalSize, 0, args);
    }

    /**
     * Dispatch with an explicit work-group size ({@code 0} = let the driver
     * choose). Required by kernels whose algorithm depends on the group
     * size -- tree reductions that size their scratch and partials buffer
     * from it. {@code globalSize} must then be a multiple of
     * {@code localSize}.
     */
    void use(int globalSize, int localSize, GpuResource... args);

    /** Blocking variant of {@link #use(int, int, GpuResource...)}. */
    void useBlocking(int globalSize, int localSize, GpuResource... args);

    /**
     * Largest work-group size this particular kernel can be launched with on
     * the bound device. Can be lower than the device-wide
     * {@link GpuExecutor#maxWorkGroupSize()} for register-heavy kernels, so
     * reduction code should clamp against this rather than guess.
     */
    int maxLocalSize();
}