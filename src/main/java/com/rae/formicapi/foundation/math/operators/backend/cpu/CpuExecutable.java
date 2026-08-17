package com.rae.formicapi.foundation.math.operators.backend.cpu;

import com.rae.formicapi.foundation.math.operators.vectors.Vector;

public abstract class CpuExecutable implements Vector {

    /** Optional. When null, every operation falls back to a plain serial loop. */
    protected CpuExecutor executor;

    /**
     * Per-vector override for the parallel dispatch threshold. -1 (default)
     * means "defer to whatever the attached executor is configured with" via
     * {@link CpuExecutor#getParallelThreshold()} - which itself defaults to
     * {@link CpuExecutor#DEFAULT_PARALLEL_THRESHOLD} unless the executor was
     * tuned with {@link CpuExecutor#calibrateThreshold(int)}. Set this only
     * if a particular vector's workload genuinely behaves differently from
     * the executor's general-purpose calibration.
     */
    protected int parallelThresholdOverride = -1;
    /** Override the parallel dispatch threshold for this vector specifically. Pass -1 to clear the override. */
    public void setParallelThreshold(int threshold) {
        this.parallelThresholdOverride = threshold;
    }


    /**
     * Attach an executor to enable parallel execution for large vectors.
     */
    public void setExecutor(CpuExecutor executor) {
        this.executor = executor;
    }

    protected boolean useParallel() {
        if (executor == null)
            return false;

        int threshold = parallelThresholdOverride >= 0
                ? parallelThresholdOverride
                : executor.getParallelThreshold();

        return this.size() >= threshold;
    }
}
