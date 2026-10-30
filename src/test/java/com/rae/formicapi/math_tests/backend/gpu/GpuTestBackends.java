package com.rae.formicapi.math_tests.backend.gpu;

import com.rae.formicapi.foundation.math.operators.backend.gpu.*;

import com.rae.formicapi.foundation.math.operators.backend.gpu.opencl.OpenCLGpuExecutor;
import com.rae.formicapi.foundation.math.operators.backend.gpu.vulkan.VulkanGpuExecutor;
import org.jetbrains.annotations.Nullable;

import java.util.function.Supplier;

/**
 * Opens executors for tests without making the build depend on having a GPU.
 *
 * <p>If the API/driver/device isn't there (no Vulkan loader, no device, missing LWJGL natives...)
 * {@link #vulkan()} / {@link #opencl()} return {@code null} and the tests skip. To make a CI job that
 * is <em>supposed</em> to have a device (e.g. Mesa lavapipe) fail instead of silently skipping, run with
 * {@code -Dformicapi.test.requireGpu=true}.
 *
 * <p>A device does <b>not</b> need fp64 to be usable: the float classes run everywhere. Tests that need double
 * precision gate on {@link #supportsDouble} (or {@link GpuTestPrecision#available}) and skip on devices without it.
 * (The OpenCL executor still insists on fp64 when it is constructed, so on OpenCL this is always true.)
 *
 * <p>This is the single place that decides "unavailable vs. broken"; the Vulkan-specific
 * {@code VulkanTestSupport} delegates here rather than repeating the logic.
 */
public final class GpuTestBackends {

    /** A kernel that does nothing but declare it needs fp64; whether the executor {@code supports} it is the probe. */
    private static final PortableKernel FP64_PROBE = new PortableKernel("test_fp64_probe")
            .opencl("__kernel void test_fp64_probe(__global double* d) { }", "cl_khr_fp64")
            .glsl("layout(std430, binding = 0) buffer D { double d[]; };\nvoid main() { }", "shaderFloat64");

    private GpuTestBackends() {
    }

    public static @Nullable GpuExecutor vulkan() {
        return open("Vulkan", VulkanGpuExecutor::new);
    }

    public static @Nullable GpuExecutor opencl() {
        return open("OpenCL", OpenCLGpuExecutor::new);
    }

    /** True if {@code ex} can run double-precision kernels. Float kernels never need this. */
    public static boolean supportsDouble(GpuExecutor ex) {
        return ex.supports(FP64_PROBE);
    }

    /** An open executor of the type {@code factory} makes, or {@code null} if this API can't be used here. */
    public static <T extends GpuExecutor> @Nullable T open(String api, Supplier<T> factory) {
        try {
            return factory.get();
        } catch (Throwable t) { // includes UnsatisfiedLinkError / NoClassDefFoundError for missing natives
            // a device that opens but computes wrongly is a real failure, not "unavailable"
            if (t instanceof IllegalStateException && String.valueOf(t.getMessage()).contains("self-test"))
                throw (IllegalStateException) t;
            if (Boolean.getBoolean("formicapi.test.requireGpu"))
                throw new AssertionError(api + " is required (formicapi.test.requireGpu=true) but could not be opened", t);
            System.err.println("[skip] " + api + " unavailable: " + t);
            return null;
        }
    }
}