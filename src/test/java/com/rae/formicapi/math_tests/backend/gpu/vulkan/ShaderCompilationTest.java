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
import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Needs shaderc's native library but <b>no GPU</b>, so it can run on any CI machine: every kernel
 * the library declares must (a) have both an OpenCL and a GLSL source and (b) compile to SPIR-V.
 *
 * <p>Precision is checked statically too, because it decides whether a kernel can run on a device without fp64:
 * a kernel must declare the fp64 requirement exactly when it uses {@code double}, and the float classes must not
 * touch double precision at all. A double kernel that forgot its requirement would be reported as supported and then
 * fail to compile on such a device instead of being refused cleanly.
 */
class ShaderCompilationTest {

    private static final int SPIRV_MAGIC = 0x07230203;

    /** Classes whose kernels work on any element type (or none): they never need fp64. */
    private static final Class<?>[] NEUTRAL = {GpuBooleanVector.class, GpuIntegerVector.class};
    private static final Class<?>[] DOUBLE_SIDE = {GpuDoubleVector.class,
            GpuPaddedCSRMatrixD.class, GpuPaddedCSRTensorD.class, GpuPaddedCSR3TensorD.class};
    private static final Class<?>[] FLOAT_SIDE = {GpuFloatVector.class,
            GpuPaddedCSRMatrixF.class, GpuPaddedCSRTensorF.class, GpuPaddedCSR3TensorF.class};

    private static final Pattern DOUBLE_USE = Pattern.compile("\\bdouble\\b|\\d(?:LF|lf)\\b");

    private static List<PortableKernel> kernelsOf(Class<?>... classes) throws IllegalAccessException {
        List<PortableKernel> out = new ArrayList<>();
        for (Class<?> c : classes) {
            for (Field f : c.getDeclaredFields()) {
                if (f.getType() == PortableKernel.class && Modifier.isStatic(f.getModifiers())) {
                    f.setAccessible(true);
                    out.add((PortableKernel) f.get(null));
                }
            }
        }
        return out;
    }

    private static List<PortableKernel> allKernels() throws IllegalAccessException {
        List<PortableKernel> out = new ArrayList<>(kernelsOf(NEUTRAL));
        out.addAll(kernelsOf(DOUBLE_SIDE));
        out.addAll(kernelsOf(FLOAT_SIDE));
        return out;
    }

    private static boolean mentionsFp64(String[] requirements) {
        return Arrays.stream(requirements).anyMatch(r -> r.toLowerCase().contains("fp64") || r.contains("Float64"));
    }

    @Test
    void thereAreKernelsToCheck() throws Exception {
        assertTrue(allKernels().size() >= 30, "found " + allKernels().size());
        assertFalse(kernelsOf(FLOAT_SIDE).isEmpty(), "the float classes must be part of the check");
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

    /**
     * Names must be unique across <em>all</em> kernels, precisions included: the OpenCL profiler aggregates timings
     * by kernel name, and error messages identify a kernel by it. A float kernel sharing the name of its double twin
     * would silently merge their statistics, so the twins are told apart by a suffix ({@code axpy} / {@code axpy_float}).
     */
    @Test
    void kernelNamesAreUnique() throws Exception {
        List<String> seen = new ArrayList<>();
        for (PortableKernel k : allKernels()) {
            assertFalse(seen.contains(k.name()), "duplicate kernel name " + k.name());
            seen.add(k.name());
        }
    }

    @Test
    void openclEntryPointNameMatchesTheKernelName() throws Exception {
        // OpenCL looks the kernel up by name, so the __kernel function must carry exactly that name
        for (PortableKernel k : allKernels())
            assertTrue(k.openclSource().contains("__kernel void " + k.name() + "("),
                    k.name() + ": no '__kernel void " + k.name() + "(' in its OpenCL source");
    }

    @Test
    void floatKernelsNeedNoDoublePrecision() throws Exception {
        for (PortableKernel k : kernelsOf(FLOAT_SIDE)) {
            assertFalse(mentionsFp64(k.glslRequirements()), k.name() + " requires fp64 in GLSL");
            assertFalse(mentionsFp64(k.openclRequirements()), k.name() + " requires fp64 in OpenCL");
            assertFalse(DOUBLE_USE.matcher(k.glslSource()).find(), k.name() + " uses double in GLSL");
            assertFalse(DOUBLE_USE.matcher(k.openclSource()).find(), k.name() + " uses double in OpenCL");
        }
    }

    @Test
    void neutralKernelsNeedNoDoublePrecision() throws Exception {
        for (PortableKernel k : kernelsOf(NEUTRAL)) {
            assertFalse(mentionsFp64(k.glslRequirements()) || mentionsFp64(k.openclRequirements()), k.name() + " requires fp64");
            assertFalse(DOUBLE_USE.matcher(k.glslSource()).find() || DOUBLE_USE.matcher(k.openclSource()).find(),
                    k.name() + " uses double");
        }
    }

    @Test
    void doubleKernelsDeclareFp64ExactlyWhenTheyUseIt() throws Exception {
        for (PortableKernel k : kernelsOf(DOUBLE_SIDE)) {
            assertEquals(DOUBLE_USE.matcher(k.glslSource()).find(), mentionsFp64(k.glslRequirements()),
                    k.name() + ": GLSL fp64 requirement doesn't match its use of double");
            assertEquals(DOUBLE_USE.matcher(k.openclSource()).find(), mentionsFp64(k.openclRequirements()),
                    k.name() + ": OpenCL fp64 requirement doesn't match its use of double");
        }
    }

    @Test
    void unknownFeatureNameInGlslRequirementsFailsLoudly() {
        assertThrows(IllegalArgumentException.class, () -> VulkanKernel.wrap("void main() {}", new String[]{"nope"}));
    }
}