package com.rae.formicapi.math_tests.backend.gpu.vulkan;

import com.rae.formicapi.foundation.math.operators.backend.gpu.*;
import com.rae.formicapi.foundation.math.operators.backend.gpu.vulkan.*;

import org.jetbrains.annotations.Nullable;

final class VulkanTestSupport {

    private VulkanTestSupport() {
    }

    /** An open executor, or {@code null} if Vulkan can't be used here (see {@code formicapi.test.requireGpu}). */
    static @Nullable VulkanGpuExecutor open() {
        try {
            return new VulkanGpuExecutor();
        } catch (Throwable t) {
            if (t instanceof IllegalStateException && String.valueOf(t.getMessage()).contains("self-test"))
                throw (IllegalStateException) t; // opens but computes wrongly: a real failure, not "unavailable"
            if (Boolean.getBoolean("formicapi.test.requireGpu"))
                throw new AssertionError("Vulkan is required (formicapi.test.requireGpu=true) but could not be opened", t);
            System.err.println("[skip] Vulkan unavailable: " + t);
            return null;
        }
    }
}