package com.rae.formicapi.foundation.math.operators.backend.gpu;

import org.jetbrains.annotations.Nullable;

/**
 * Base class for every GPU-backed vector implementation. Mirrors
 * {@link com.rae.formicapi.foundation.math.operators.backend.cpu.CpuExecutable}:
 * where that class holds an optional {@code CpuExecutor} and falls back to a
 * plain serial loop when none is attached, this class holds a
 * {@link GpuExecutor} that is <b>required</b> - there is no "serial on GPU"
 * fallback, so every operation needs one attached to actually dispatch a
 * kernel through.
 *
 * <p>Unlike {@code CpuExecutable}'s {@code useParallel()} threshold check,
 * there's no size-based decision to make here: once a value lives in a
 * {@code Gpu*Vector} at all, every operation on it goes through the
 * executor. The CPU/GPU split itself - which backend a given piece of data
 * lives in - is the caller's decision, made once when the vector is
 * created.
 *
 * <p>Deliberately does not declare {@code implements Vector} the way
 * {@code CpuExecutable} does: each concrete {@code Gpu*Vector} already gets
 * {@code Vector} transitively through its own interface
 * ({@code DoubleVector}/{@code BooleanVector}/{@code IntegerVector} extend
 * it), so a second, separate {@code implements Vector} here wouldn't add
 * any capability - nothing needs to hold a bare {@code GpuExecutable}
 * reference and call {@code Vector} methods on it. {@code CpuExecutable}'s
 * copy of this looks like the same redundancy; not changed here since that
 * class is out of scope for this GPU work.
 */
public abstract class GpuExecutable {

    protected @Nullable GpuExecutor executor;

    public @Nullable GpuExecutor getExecutor() {
        return executor;
    }

    public void setExecutor(@Nullable GpuExecutor executor) {
        this.executor = executor;
    }

    protected GpuExecutor requireExecutor() {
        if (executor == null)
            throw new IllegalStateException(
                    getClass().getSimpleName() + " has no GpuExecutor attached - call setExecutor(...) before use");
        return executor;
    }
}