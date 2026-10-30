package com.rae.formicapi.math_tests.backend.gpu.vulkan;

import com.rae.formicapi.foundation.math.operators.backend.gpu.vulkan.*;
import com.rae.formicapi.math_tests.backend.gpu.GpuTestBackends;

import org.jetbrains.annotations.Nullable;

final class VulkanTestSupport {

    private VulkanTestSupport() {
    }

    /**
     * An open executor, or {@code null} if Vulkan can't be used here (see {@code formicapi.test.requireGpu}).
     * Typed as {@link VulkanGpuExecutor} for the tests that need its device info; the "unavailable vs. broken"
     * decision lives in {@link GpuTestBackends#open}.
     */
    static @Nullable VulkanGpuExecutor open() {
        return GpuTestBackends.open("Vulkan", VulkanGpuExecutor::new);
    }
}