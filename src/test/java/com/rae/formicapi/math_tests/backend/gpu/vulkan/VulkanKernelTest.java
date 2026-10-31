package com.rae.formicapi.math_tests.backend.gpu.vulkan;

import com.rae.formicapi.foundation.math.operators.backend.gpu.PortableKernel;
import com.rae.formicapi.foundation.math.operators.backend.gpu.vulkan.VulkanGpuExecutor;
import com.rae.formicapi.foundation.math.operators.backend.gpu.vulkan.VulkanKernel;
import com.rae.formicapi.math_tests.backend.gpu.GpuTestBackends;
import org.junit.jupiter.api.*;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Kernel creation, support checks and error reporting. The kernels here are float on purpose: none of these
 * properties has anything to do with double precision, and a float kernel is accepted by every device, so the tests run
 * everywhere instead of failing on a device without fp64 for a reason unrelated to what they check. Double precision has
 * exactly one test of its own, {@link #doubleKernelSupportFollowsTheDevice}.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class VulkanKernelTest {

    private static final String TRIVIAL = """
            layout(std430, binding = 0) buffer D { float d[]; };
            layout(push_constant) uniform PC { int gsz; } pc;
            void main() { if (GID < uint(pc.gsz)) d[GID] = 1.0; }
            """;

    private VulkanGpuExecutor ex;

    @BeforeAll
    void open() {
        ex = VulkanTestSupport.open();
        assumeTrue(ex != null, "Vulkan not available");
    }

    @AfterAll
    void close() {
        if (ex != null) ex.close();
    }

    @Test
    void kernelWithoutGlslSourceIsUnsupported() {
        PortableKernel openclOnly = new PortableKernel("test_cl_only").opencl("__kernel void test_cl_only() {}");
        assertFalse(ex.supports(openclOnly));
        assertThrows(UnsupportedOperationException.class, () -> openclOnly.bind(ex));
    }

    @Test
    void unknownFeatureRequirementIsUnsupported() {
        PortableKernel k = new PortableKernel("test_feature").glsl(TRIVIAL, "someFeatureNobodyHas");
        assertFalse(ex.supports(k));
        assertThrows(UnsupportedOperationException.class, () -> k.bind(ex));
    }

    @Test
    void supportedKernelReportsSupported() {
        assertTrue(ex.supports(new PortableKernel("test_ok").glsl(TRIVIAL, "shaderFloat32")));
    }

    /**
     * fp64 is optional: a kernel that needs it must be reported as unsupported and refused with
     * {@link UnsupportedOperationException} on a device without it, and must bind normally on a device with it. Either way
     * {@code supports} and {@code bind} must agree (never "supported" and then a compile failure).
     */
    @Test
    void doubleKernelSupportFollowsTheDevice() {
        PortableKernel k = new PortableKernel("test_double").glsl("""
                layout(std430, binding = 0) buffer D { double d[]; };
                layout(push_constant) uniform PC { int gsz; } pc;
                void main() { if (GID < uint(pc.gsz)) d[GID] = 1.0LF; }
                """, "shaderFloat64");
        if (GpuTestBackends.supportsDouble(ex)) {
            assertTrue(ex.supports(k));
            assertDoesNotThrow(() -> k.bind(ex));
        } else {
            assertFalse(ex.supports(k));
            assertThrows(UnsupportedOperationException.class, () -> k.bind(ex));
        }
    }

    @Test
    void compileErrorNamesTheKernelAndShowsTheCompilerMessage() {
        PortableKernel bad = new PortableKernel("test_broken").glsl("""
                layout(std430, binding = 0) buffer D { float d[]; };
                void main() { this_function_does_not_exist(); }
                """, "shaderFloat32");
        RuntimeException e = assertThrows(RuntimeException.class, () -> bad.bind(ex));
        assertTrue(e.getMessage().contains("test_broken"), e.getMessage());
    }

    @Test
    void kernelWithNoBufferBindingsIsRejected() {
        PortableKernel none = new PortableKernel("test_nobuf").glsl("void main() {}", "shaderFloat32");
        assertThrows(IllegalArgumentException.class, () -> none.bind(ex));
    }

    @Test
    void kernelWithTooManyBindingsIsRejected() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i <= VulkanKernel.MAX_BINDINGS; i++)
            sb.append("layout(std430, binding = ").append(i).append(") buffer B").append(i).append(" { float b").append(i).append("[]; };\n");
        sb.append("void main() {}\n");
        PortableKernel many = new PortableKernel("test_many").glsl(sb.toString(), "shaderFloat32");
        assertThrows(IllegalArgumentException.class, () -> many.bind(ex));
    }

    @Test
    void rebindingTheSameExecutorReusesTheCompiledKernel() {
        PortableKernel k = new PortableKernel("test_rebind").glsl(TRIVIAL, "shaderFloat32");
        k.bind(ex);
        assertDoesNotThrow(() -> k.bind(ex));
        assertTrue(k.maxLocalSize() >= 128);
    }

    @Test
    void nativeKernelRefusesAForeignExecutor() {
        VulkanGpuExecutor other = VulkanTestSupport.open();
        assumeTrue(other != null);
        try {
            VulkanKernel k = (VulkanKernel) ex.createKernel(new PortableKernel("test_foreign").glsl(TRIVIAL, "shaderFloat32"));
            assertThrows(IllegalArgumentException.class, () -> k.bind(other));
        } finally {
            other.close();
        }
    }
}