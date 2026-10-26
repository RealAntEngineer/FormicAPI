package com.rae.formicapi.math_tests.backend.gpu;

import com.rae.formicapi.foundation.math.operators.backend.gpu.GpuExecutor;

/** The shared GPU contract, run on the Vulkan backend. */
class VulkanContractTest extends GpuContractTest {
    @Override
    public GpuExecutor open() {
        return GpuTestBackends.vulkan();
    }
}
