package com.rae.formicapi.math_tests.backend.gpu.vulkan;

import com.rae.formicapi.foundation.math.operators.backend.gpu.*;
import com.rae.formicapi.foundation.math.operators.backend.gpu.vulkan.VulkanKernel;
import org.junit.jupiter.api.Test;
import org.lwjgl.system.MemoryUtil;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Needs shaderc's native library but <b>no GPU</b>, so it can run on any CI machine: every kernel
 * the library declares must (a) have both an OpenCL and a GLSL source and (b) compile to SPIR-V.
 */
class ShaderCompilationTest {

    private static final int SPIRV_MAGIC = 0x07230203;

    private static List<PortableKernel> allKernels() throws IllegalAccessException {
        List<PortableKernel> out = new ArrayList<>();
        for (Class<?> c : new Class<?>[]{GpuDoubleVector.class, GpuBooleanVector.class, GpuIntegerVector.class,
                GpuPaddedCSRMatrixD.class, GpuPaddedCSRTensorD.class, GpuPaddedCSR3TensorD.class}) {
            for (Field f : c.getDeclaredFields()) {
                if (f.getType() == PortableKernel.class && Modifier.isStatic(f.getModifiers())) {
                    f.setAccessible(true);
                    out.add((PortableKernel) f.get(null));
                }
            }
        }
        return out;
    }

    @Test
    void thereAreKernelsToCheck() throws Exception {
        assertTrue(allKernels().size() >= 15, "found " + allKernels().size());
    }

    @Test
    void everyKernelHasBothBackendSources() throws Exception {
        for (PortableKernel k : allKernels()) {
            assertNotNull(k.openclSource(), k.name() + " has no OpenCL source");
            assertNotNull(k.glslSource(), k.name() + " has no GLSL source");
            assertFalse(k.glslSource().contains("#version"), k.name() + ": the backend adds #version, kernels must not");
        }
    }

    @Test
    void everyGlslKernelCompilesToSpirv() throws Exception {
        for (PortableKernel k : allKernels()) {
            ByteBuffer spirv = VulkanKernel.compile(k.name(), VulkanKernel.wrap(k.glslSource(), k.glslRequirements()));
            try {
                assertTrue(spirv.remaining() >= 20 && spirv.remaining() % 4 == 0, k.name() + ": implausible SPIR-V size");
                assertEquals(SPIRV_MAGIC, spirv.order(ByteOrder.nativeOrder()).getInt(0), k.name() + ": bad SPIR-V magic");
            } finally {
                MemoryUtil.memFree(spirv);
            }
        }
    }

    @Test
    void kernelNamesAreUnique() throws Exception {
        List<String> seen = new ArrayList<>();
        for (PortableKernel k : allKernels()) {
            assertFalse(seen.contains(k.name()), "duplicate kernel name " + k.name());
            seen.add(k.name());
        }
    }

    @Test
    void unknownFeatureNameInGlslRequirementsFailsLoudly() {
        assertThrows(IllegalArgumentException.class, () -> VulkanKernel.wrap("void main() {}", new String[]{"nope"}));
    }
}