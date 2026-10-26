package com.rae.formicapi.math_tests.backend.gpu;

import com.rae.formicapi.foundation.math.operators.backend.gpu.PortableKernel;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/** Pure-Java checks of {@link PortableKernel}; no GPU, no native libraries. */
class PortableKernelTest {

    @Test
    void builderStoresBothSourcesAndRequirements() {
        PortableKernel k = new PortableKernel("k")
                .opencl("cl src", "cl_khr_fp64")
                .glsl("glsl src", "shaderFloat64", "shaderInt64");
        assertEquals("k", k.name());
        assertEquals("cl src", k.openclSource());
        assertArrayEquals(new String[]{"cl_khr_fp64"}, k.openclRequirements());
        assertEquals("glsl src", k.glslSource());
        assertArrayEquals(new String[]{"shaderFloat64", "shaderInt64"}, k.glslRequirements());
    }

    @Test
    void sourcesAreOptional() {
        PortableKernel k = new PortableKernel("only-gl").glsl("x");
        assertNull(k.openclSource());
        assertEquals(0, k.openclRequirements().length);
    }

    @Test
    void usingAnUnboundKernelFailsClearly() {
        PortableKernel k = new PortableKernel("unbound");
        IllegalStateException e = assertThrows(IllegalStateException.class, () -> k.use(1));
        assertTrue(e.getMessage().contains("unbound"));
        assertThrows(IllegalStateException.class, () -> k.useBlocking(1));
        assertThrows(IllegalStateException.class, k::maxLocalSize);
    }
}