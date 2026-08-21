package com.rae.formicapi.foundation.math.operators.backend.cpu;

import com.rae.formicapi.foundation.math.operators.vectors.Vector;
import org.jetbrains.annotations.Nullable;

public abstract class CpuExecutable {

    /** Optional. When null, every operation falls back to a plain serial loop. */
    protected @Nullable CpuExecutor executor;

    /**
     * Attach an executor to enable parallel execution for large vectors.
     */
    public void setExecutor(@Nullable CpuExecutor executor) {
        this.executor = executor;
    }
}
