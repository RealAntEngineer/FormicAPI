package com.rae.formicapi.math_tests.backend.gpu;

import com.rae.formicapi.foundation.math.operators.backend.gpu.GpuExecutor;

/** The same contract on OpenCL -- this is what exercises the newly implemented OpenCL operations. */
class OpenCLContractTest extends GpuContractTest {
    @Override
    GpuExecutor open() {
        return GpuTestBackends.opencl();
    }
}