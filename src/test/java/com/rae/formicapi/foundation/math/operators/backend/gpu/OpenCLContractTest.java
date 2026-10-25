package com.rae.formicapi.foundation.math.operators.backend.gpu;

/** The same contract on OpenCL -- this is what exercises the newly implemented OpenCL operations. */
class OpenCLContractTest extends GpuContractTest {
    @Override
    protected GpuExecutor open() {
        return GpuTestBackends.opencl();
    }
}
