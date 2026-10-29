package com.rae.formicapi.math_tests.backend.gpu.vulkan;

import com.rae.formicapi.foundation.math.operators.backend.gpu.GpuResource;
import com.rae.formicapi.foundation.math.operators.backend.gpu.PortableKernel;
import com.rae.formicapi.foundation.math.operators.backend.gpu.ResourceEvent;
import com.rae.formicapi.foundation.math.operators.backend.gpu.vulkan.VulkanGpuExecutor;
import org.junit.jupiter.api.*;

import java.util.Arrays;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class VulkanGpuExecutorTest {

    /** {@code out[i] = in[i] * a + n} -- scalars in (int, double) order, so the double needs 4 bytes of padding. */
    private static final PortableKernel INT_THEN_DOUBLE = new PortableKernel("test_int_then_double").glsl("""
            layout(std430, binding = 0) buffer I { double in_[]; };
            layout(std430, binding = 1) buffer O { double res[]; };
            layout(push_constant) uniform PC { int gsz; int n; double a; } pc;
            void main() {
                uint i = GID;
                if (i >= uint(pc.gsz)) return;
                res[i] = in_[i] * pc.a + double(pc.n);
            }
            """, "shaderFloat64");

    /** Same maths, scalars in (double, int) order: double directly after gsz leaves a 4-byte hole. */
    private static final PortableKernel DOUBLE_THEN_INT = new PortableKernel("test_double_then_int").glsl("""
            layout(std430, binding = 0) buffer I { double in_[]; };
            layout(std430, binding = 1) buffer O { double res[]; };
            layout(push_constant) uniform PC { int gsz; double a; int n; } pc;
            void main() {
                uint i = GID;
                if (i >= uint(pc.gsz)) return;
                res[i] = in_[i] * pc.a + double(pc.n);
            }
            """, "shaderFloat64");

    /** {@code d[0] += 1}, one invocation: used to pile up thousands of dependent dispatches. */
    private static final PortableKernel INCREMENT = new PortableKernel("test_increment").glsl("""
            layout(std430, binding = 0) buffer D { double d[]; };
            layout(push_constant) uniform PC { int gsz; } pc;
            void main() {
                if (GID >= uint(pc.gsz)) return;
                d[GID] += 1.0LF;
            }
            """, "shaderFloat64");

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

    // ------------------------------------------------------------------ device info

    @Test
    void reportsDeviceAndLimits() {
        assertFalse(ex.deviceName().isBlank());
        assertTrue(ex.maxWorkGroupSize() >= 128, "Vulkan guarantees at least 128 invocations per group");
        assertFalse(VulkanGpuExecutor.deviceNames().isEmpty());
    }

    @Test
    void rejectsOutOfRangeDeviceIndex() {
        assertThrows(IllegalArgumentException.class, () -> new VulkanGpuExecutor(10_000));
    }

    // ------------------------------------------------------------------ transfers

    private double[] randomDoubles(int n, long seed) {
        Random r = new Random(seed);
        double[] a = new double[n];
        for (int i = 0; i < n; i++) a[i] = r.nextGaussian();
        return a;
    }

    @Test
    void doubleTransfersWithOffsets() {
        GpuResource buf = ex.allocateDoubleBuffer(100);
        double[] data = randomDoubles(100, 1);
        ex.uploadDoubles(buf, data, 100);

        double[] part = new double[10];
        ex.downloadDoubles(buf, 10, part, 10);
        assertArrayEquals(Arrays.copyOfRange(data, 10, 20), part);

        double[] patch = {-1, -2, -3};
        ex.uploadDoubles(buf, 50, patch, 0, 3);
        ex.uploadDoubles(buf, 60, new double[]{9, 8, 7, 6}, 2, 2); // host offset

        double[] expected = data.clone();
        System.arraycopy(patch, 0, expected, 50, 3);
        expected[60] = 7;
        expected[61] = 6;
        double[] all = new double[100];
        ex.downloadDoubles(buf, all, 100);
        assertArrayEquals(expected, all);
        buf.release();
    }

    @Test
    void singleElementAccessors() {
        GpuResource d = ex.allocateDoubleBuffer(4), i = ex.allocateIntBuffer(4), b = ex.allocateByteBuffer(7);
        ex.uploadDoubleAt(d, 3, Math.PI);
        ex.uploadIntAt(i, 2, -123456);
        ex.uploadByteAt(b, 5, (byte) -7);
        assertEquals(Math.PI, ex.downloadDoubleAt(d, 3));
        assertEquals(-123456, ex.downloadIntAt(i, 2));
        assertEquals((byte) -7, ex.downloadByteAt(b, 5));
        d.release(); i.release(); b.release();
    }

    @Test
    void intAndByteTransfers() {
        int[] ints = new Random(2).ints(1000).toArray();
        GpuResource ib = ex.allocateIntBuffer(1000);
        ex.uploadInts(ib, ints, 1000);
        int[] backInts = new int[1000];
        ex.downloadInts(ib, backInts, 1000);
        assertArrayEquals(ints, backInts);

        byte[] bytes = new byte[1001]; // not a multiple of 4
        new Random(3).nextBytes(bytes);
        GpuResource bb = ex.allocateByteBuffer(1001);
        ex.uploadBytes(bb, bytes, 1001);
        byte[] backBytes = new byte[1001];
        ex.downloadBytes(bb, backBytes, 1001);
        assertArrayEquals(bytes, backBytes);
        ib.release(); bb.release();
    }

    @Test
    void transferLargerThanInitialStagingGrowsIt() {
        int n = 3_000_000; // 24 MB vs 1 MiB initial staging
        double[] data = randomDoubles(n, 4);
        GpuResource buf = ex.allocateDoubleBuffer(n);
        ex.uploadDoubles(buf, data, n);
        double[] back = new double[n];
        ex.downloadDoubles(buf, back, n);
        assertArrayEquals(data, back);
        buf.release();
    }

    @Test
    @Tag("slow")
    void transferLargerThanMaxStagingIsChunked() {
        int n = 9_000_000; // 72 MB > the 64 MiB staging cap -> two chunks
        double[] data = randomDoubles(n, 5);
        GpuResource buf = ex.allocateDoubleBuffer(n);
        ex.uploadDoubles(buf, data, n);
        double[] back = new double[n];
        ex.downloadDoubles(buf, back, n);
        assertArrayEquals(data, back);
        buf.release();
    }

    // ------------------------------------------------------------------ copy

    @Test
    void copyWithOffsets() {
        double[] data = randomDoubles(50, 6);
        GpuResource src = ex.allocateDoubleBuffer(50), dst = ex.allocateDoubleBuffer(50);
        ex.uploadDoubles(src, data, 50);
        ex.fillDoubleBuffer(dst, 0.0, 50);
        ex.copyDoubleBuffer(src, 10, dst, 30, 15);

        double[] expected = new double[50];
        System.arraycopy(data, 10, expected, 30, 15);
        double[] back = new double[50];
        ex.downloadDoubles(dst, back, 50);
        assertArrayEquals(expected, back);
        src.release(); dst.release();
    }

    @Test
    void copyWithinHandlesOverlap() {
        for (int[] c : new int[][]{{0, 4, 12}, {4, 0, 12}, {0, 20, 10}, {7, 7, 5}}) {
            double[] data = randomDoubles(30, 7);
            GpuResource buf = ex.allocateDoubleBuffer(30);
            ex.uploadDoubles(buf, data, 30);
            ex.copyDoubleWithin(buf, c[0], c[1], c[2]);

            double[] expected = data.clone();
            System.arraycopy(expected, c[0], expected, c[1], c[2]);
            double[] back = new double[30];
            ex.downloadDoubles(buf, back, 30);
            assertArrayEquals(expected, back, Arrays.toString(c));
            buf.release();
        }
    }

    // ------------------------------------------------------------------ fills

    @Test
    void doubleFills() {
        GpuResource buf = ex.allocateDoubleBuffer(20);
        double[] back = new double[20];

        ex.fillDoubleBuffer(buf, 0.0, 20); // zero: single 32-bit pattern path
        ex.downloadDoubles(buf, back, 20);
        assertArrayEquals(new double[20], back);

        ex.fillDoubleBuffer(buf, 4, 3.25, 10); // arbitrary value with an offset: compute-shader path
        ex.downloadDoubles(buf, back, 20);
        double[] expected = new double[20];
        Arrays.fill(expected, 4, 14, 3.25);
        assertArrayEquals(expected, back);
        buf.release();
    }

    @Test
    void intFillsWithOffset() {
        GpuResource buf = ex.allocateIntBuffer(10);
        ex.fillIntBuffer(buf, 0, 10);
        ex.fillIntBuffer(buf, 3, -5, 4);
        int[] back = new int[10];
        ex.downloadInts(buf, back, 10);
        assertArrayEquals(new int[]{0, 0, 0, -5, -5, -5, -5, 0, 0, 0}, back);
        buf.release();
    }

    @Test
    void byteFillIsExactForEveryOffsetAndLength() {
        // vkCmdFillBuffer works on aligned 4-byte words; head/tail bytes take another path. Check all alignments.
        final int size = 32;
        GpuResource buf = ex.allocateByteBuffer(size);
        for (int offset = 0; offset <= 8; offset++) {
            for (int length = 0; length <= 14; length++) {
                byte[] base = new byte[size];
                Arrays.fill(base, (byte) 0x55);
                ex.uploadBytes(buf, base, size);

                ex.fillByteBuffer(buf, offset, (byte) 0xA7, length);

                byte[] expected = base.clone();
                Arrays.fill(expected, offset, offset + length, (byte) 0xA7);
                byte[] back = new byte[size];
                ex.downloadBytes(buf, back, size);
                assertArrayEquals(expected, back, "offset=" + offset + " length=" + length);
            }
        }
        buf.release();
    }

    // ------------------------------------------------------------------ resource lifetime

    @Test
    void releasingRightAfterEnqueueingWorkIsSafe() {
        // the copy may still be queued when release() is called; destruction must wait for it
        for (int round = 0; round < 200; round++) {
            GpuResource src = ex.allocateDoubleBuffer(1000), dst = ex.allocateDoubleBuffer(1000);
            ex.fillDoubleBuffer(src, 1000.0 + round, 1000);
            ex.copyDoubleBuffer(src, dst, 1000);
            src.release();

            assertEquals(1000.0 + round, ex.downloadDoubleAt(dst, 999));
            dst.release();
        }
    }

    @Test
    void doubleReleaseIsHarmless() {
        GpuResource buf = ex.allocateIntBuffer(4);
        buf.release();
        assertDoesNotThrow(buf::release);
    }

    @Test
    void manyLiveBuffers() {
        GpuResource[] bufs = new GpuResource[300];
        for (int i = 0; i < bufs.length; i++) {
            bufs[i] = ex.allocateIntBuffer(16);
            ex.fillIntBuffer(bufs[i], i, 16);
        }
        for (int i = 0; i < bufs.length; i++) {
            assertEquals(i, ex.downloadIntAt(bufs[i], 15));
            bufs[i].release();
        }
    }

    // ------------------------------------------------------------------ dispatch

    @Test
    void scalarPushConstantsAreAlignedInEitherOrder() {
        int n = 1000;
        double[] in = randomDoubles(n, 8);
        GpuResource inBuf = ex.allocateDoubleBuffer(n), outBuf = ex.allocateDoubleBuffer(n);
        ex.uploadDoubles(inBuf, in, n);
        double[] expected = new double[n];
        for (int i = 0; i < n; i++) expected[i] = in[i] * 2.5 + 7;

        INT_THEN_DOUBLE.bind(ex);
        INT_THEN_DOUBLE.useBlocking(n, inBuf, outBuf, ex.scalar(7), ex.scalar(2.5));
        double[] out = new double[n];
        ex.downloadDoubles(outBuf, out, n);
        assertArrayEquals(expected, out, 1e-12);

        ex.fillDoubleBuffer(outBuf, 0.0, n);
        DOUBLE_THEN_INT.bind(ex);
        DOUBLE_THEN_INT.useBlocking(n, inBuf, outBuf, ex.scalar(2.5), ex.scalar(7));
        ex.downloadDoubles(outBuf, out, n);
        assertArrayEquals(expected, out, 1e-12);
        inBuf.release(); outBuf.release();
    }

    @Test
    void sizeNotMultipleOfGroupDoesNotWritePastTheEnd() {
        // 1000 isn't a multiple of the default group size; the guard must stop the extra invocations
        int n = 1000;
        GpuResource in = ex.allocateDoubleBuffer(n), out = ex.allocateDoubleBuffer(n + 64);
        ex.fillDoubleBuffer(in, 0.0, n);
        ex.fillDoubleBuffer(out, 99.0, n + 64);
        INT_THEN_DOUBLE.bind(ex);
        INT_THEN_DOUBLE.useBlocking(n, in, out, ex.scalar(1), ex.scalar(1.0));

        double[] back = new double[n + 64];
        ex.downloadDoubles(out, back, n + 64);
        for (int i = 0; i < n; i++) assertEquals(1.0, back[i], "element " + i);
        for (int i = n; i < n + 64; i++) assertEquals(99.0, back[i], "guard failed at " + i);
        in.release(); out.release();
    }

    @Test
    void manyUnawaitedDispatchesExhaustDescriptorPoolGracefully() {
        // far more dispatches than the 4096-set descriptor pool, none waited on individually
        GpuResource buf = ex.allocateDoubleBuffer(1);
        ex.fillDoubleBuffer(buf, 0.0, 1);
        INCREMENT.bind(ex);
        final int count = 6000;
        for (int i = 0; i < count; i++)
            INCREMENT.use(1, buf);
        assertEquals(count, ex.downloadDoubleAt(buf, 0)); // also proves they ran in order, one after another
        buf.release();
    }

    @Test
    void invalidLaunchesAreRejected() {
        GpuResource buf = ex.allocateDoubleBuffer(64);
        INCREMENT.bind(ex);
        assertThrows(IllegalArgumentException.class, () -> INCREMENT.use(100, 64, buf));          // 100 % 64 != 0
        assertThrows(IllegalArgumentException.class, () -> INCREMENT.use(1 << 20, 1 << 20, buf));  // group size over the device limit
        assertThrows(IllegalArgumentException.class, () -> INCREMENT.use(1));                      // missing buffer
        assertThrows(IllegalArgumentException.class, () -> INCREMENT.use(1, buf, buf));            // extra buffer
        assertThrows(NullPointerException.class, () -> INCREMENT.use(1, (GpuResource) null));
        assertThrows(RuntimeException.class, () -> INCREMENT.use(1, new ForeignResource()));
        buf.release();
    }

    @Test
    void zeroSizedDispatchIsANoOp() {
        GpuResource buf = ex.allocateDoubleBuffer(1);
        ex.fillDoubleBuffer(buf, 5.0, 1);
        INCREMENT.bind(ex);
        INCREMENT.use(0, buf);
        assertEquals(5.0, ex.downloadDoubleAt(buf, 0));
        buf.release();
    }

    private static final class ForeignResource implements GpuResource {
        public ResourceEvent[] pendingEvents() {
            return new ResourceEvent[0]; }
        public void recordEvent(ResourceEvent e) {

        }
        public void settle() { }
        public void release() { }
    }

    // ------------------------------------------------------------------ lifecycle

    @Test
    void finishAndCloseAreIdempotent() {
        VulkanGpuExecutor own = VulkanTestSupport.open();
        assumeTrue(own != null);
        GpuResource buf = own.allocateDoubleBuffer(8);
        own.fillDoubleBuffer(buf, 1.0, 8);
        own.finish();
        buf.release();
        own.close();
        assertDoesNotThrow(own::close);
        assertDoesNotThrow(buf::release);
    }

    @Test
    void dispatchAfterCloseFails() {
        VulkanGpuExecutor own = VulkanTestSupport.open();
        assumeTrue(own != null);
        GpuResource buf = own.allocateDoubleBuffer(1);
        PortableKernel k = new PortableKernel("test_closed").glsl(INCREMENT.glslSource(), "shaderFloat64");
        k.bind(own);
        own.close();
        assertThrows(IllegalStateException.class, () -> k.use(1, buf));
    }

    @Test
    void twoExecutorsCoexist() {
        VulkanGpuExecutor other = VulkanTestSupport.open();
        assumeTrue(other != null);
        try {
            GpuResource a = ex.allocateIntBuffer(1), b = other.allocateIntBuffer(1);
            ex.uploadIntAt(a, 0, 11);
            other.uploadIntAt(b, 0, 22);
            assertEquals(11, ex.downloadIntAt(a, 0));
            assertEquals(22, other.downloadIntAt(b, 0));
        } finally {
            other.close();
        }
    }
}