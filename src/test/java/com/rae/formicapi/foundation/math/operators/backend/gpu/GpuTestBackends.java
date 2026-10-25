package com.rae.formicapi.foundation.math.operators.backend.gpu;

import com.rae.formicapi.foundation.math.operators.backend.gpu.opencl.OpenCLGpuExecutor;
import com.rae.formicapi.foundation.math.operators.backend.gpu.vulkan.VulkanGpuExecutor;
import org.jetbrains.annotations.Nullable;

import java.util.function.Supplier;

/**
 * Opens executors for tests without making the build depend on having a GPU.
 *
 * <p>If the API/driver/device isn't there (no Vulkan loader, no fp64 device, missing LWJGL natives...)
 * {@link #vulkan()} / {@link #opencl()} return {@code null} and the tests skip. To make a CI job that
 * is <em>supposed</em> to have a device (e.g. Mesa lavapipe) fail instead of silently skipping, run with
 * {@code -Dformicapi.test.requireGpu=true}.
 */
public final class GpuTestBackends {

    private GpuTestBackends() {
    }

    public static @Nullable GpuExecutor vulkan() {
        return open("Vulkan", VulkanGpuExecutor::new);
    }

    static @Nullable GpuExecutor opencl() {
        return open("OpenCL", OpenCLGpuExecutor::new);
    }

    private static @Nullable GpuExecutor open(String api, Supplier<GpuExecutor> factory) {
        try {
            return factory.get();
        } catch (Throwable t) { // includes UnsatisfiedLinkError / NoClassDefFoundError for missing natives
            if (Boolean.getBoolean("formicapi.test.requireGpu"))
                throw new AssertionError(api + " is required (formicapi.test.requireGpu=true) but could not be opened", t);
            System.err.println("[skip] " + api + " unavailable: " + t);
            return null;
        }
    }
}
