package com.rae.formicapi.foundation.math.operators.backend.gpu;

import com.rae.formicapi.foundation.math.operators.backend.gpu.opencl.OpenCLGpuExecutor;
import org.jetbrains.annotations.Nullable;

/**
 * Base class for every GPU-backed vector implementation.
 */
public abstract class GpuExecutable {

    protected @Nullable OpenCLGpuExecutor executor;

    public @Nullable OpenCLGpuExecutor getExecutor() {
        return executor;
    }

    /**
     * Attaches {@code executor} and eagerly binds this class's kernels
     * against it (see {@link #bindKernels}), so every arithmetic method
     * can dispatch straight away with no first-use compile stall and no
     * per-call binding to worry about.
     */
    public void setExecutor(@Nullable OpenCLGpuExecutor executor) {
        this.executor = executor;
        if (executor != null)
            bindKernels(executor);
    }

    /**
     * Override to bind this class's static {@link Kernel} constants
     * against {@code executor} -- one {@code KERNEL.bind(executor)} call
     * per kernel this class dispatches (see {@code GpuDoubleVector}). Since
     * kernel constants are {@code static final} and shared across every
     * instance, re-binding here when a second vector attaches to the same
     * executor is a cheap no-op (the kernel is already compiled), not a
     * recompile. No-op by default for classes that don't dispatch kernels
     * directly.
     */
    protected void bindKernels(GpuExecutor executor) {
    }

    protected OpenCLGpuExecutor requireExecutor() {
        if (executor == null)
            throw new IllegalStateException(
                    getClass().getSimpleName() + " has no GpuExecutor attached - call setExecutor(...) before use");
        return executor;
    }
}