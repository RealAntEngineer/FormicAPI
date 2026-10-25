package com.rae.formicapi.foundation.math.operators.backend.gpu.vulkan;

import com.rae.formicapi.foundation.math.operators.backend.gpu.GpuContractTest;
import com.rae.formicapi.foundation.math.operators.backend.gpu.GpuTestBackends;
import com.rae.formicapi.foundation.math.operators.backend.gpu.GpuExecutor;

/** The shared GPU contract, run on the Vulkan backend. */
class VulkanContractTest extends GpuContractTest {
    @Override
    public GpuExecutor open() {
        return GpuTestBackends.vulkan();
    }
}
