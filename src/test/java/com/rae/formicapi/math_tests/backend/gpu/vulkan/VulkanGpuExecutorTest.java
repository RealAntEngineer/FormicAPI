package com.rae.formicapi.math_tests.backend.gpu.vulkan;

import com.rae.formicapi.foundation.math.operators.backend.gpu.GpuResource;
import com.rae.formicapi.foundation.math.operators.backend.gpu.PortableKernel;
import com.rae.formicapi.foundation.math.operators.backend.gpu.ResourceEvent;
import com.rae.formicapi.foundation.math.operators.backend.gpu.vulkan.VulkanEvent;
import com.rae.formicapi.foundation.math.operators.backend.gpu.vulkan.VulkanGpuExecutor;
import com.rae.formicapi.math_tests.backend.gpu.GpuTestPrecision;
import org.junit.jupiter.api.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Executor-level tests of the Vulkan backend: transfers, copies, fills, dispatch, resource lifetime, and the
 * per-resource event handling (ordering of dependent work without blocking the host).
 *
 * <p><b>Device requirements.</b> Nothing here needs fp64 unless it is about double precision. Tests whose subject is
 * typed element handling (transfers, accessors, copies, fills, scalar packing) run once per precision through
 * {@link #forEachPrecision}; the double flavour is skipped on a device without fp64 and the float flavour always runs.
 * Tests whose subject is the machinery (dispatch validation, lifetime, ordering, concurrency) are precision-neutral and
 * use float or int buffers, so they run on every device instead of being copied per type.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class VulkanGpuExecutorTest {

    /** {@code d[i] += 1}: used to pile up thousands of dependent dispatches. Float, so it runs on every device. */
    private static final PortableKernel INCREMENT = new PortableKernel("test_increment").glsl("""
            layout(std430, binding = 0) buffer D { float d[]; };
            layout(push_constant) uniform PC { int gsz; } pc;
            void main() {
                if (GID >= uint(pc.gsz)) return;
                d[GID] += 1.0;
            }
            """, "shaderFloat32");

    /**
     * {@code res[i] = in[i] * a + n}, with the two scalars declared in the given order. For double the value
     * needs 8-byte alignment, so one order leaves a 4-byte hole after {@code gsz} and the other doesn't; for float
     * everything is 4 bytes and packs tightly. Either way the scalars must arrive in declaration order.
     */
    private static PortableKernel scalarKernel(GpuTestPrecision p, boolean intFirst) {
        String t = p.isFloat() ? "float" : "double";
        String block = intFirst ? "int gsz; int n; " + t + " a;" : "int gsz; " + t + " a; int n;";
        return new PortableKernel("test_scalars_" + t + (intFirst ? "_int_first" : "_value_first")).glsl("""
                layout(std430, binding = 0) buffer I { %s in_[]; };
                layout(std430, binding = 1) buffer O { %s res[]; };
                layout(push_constant) uniform PC { %s } pc;
                void main() {
                    uint i = GID;
                    if (i >= uint(pc.gsz)) return;
                    res[i] = in_[i] * pc.a + %s(pc.n);
                }
                """.formatted(t, t, block, t), p.isFloat() ? "shaderFloat32" : "shaderFloat64");
    }

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

    /** One dynamic test per precision; double is skipped (not failed) on a device without fp64. */
    private Stream<DynamicTest> forEachPrecision(Consumer<GpuTestPrecision> body) {
        return Arrays.stream(GpuTestPrecision.values()).map(p -> DynamicTest.dynamicTest(p.name(), () -> {
            assumeTrue(p.available(ex), "device has no fp64, so double buffers can't be used here");
            body.accept(p);
        }));
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

    /** Gaussian values already rounded to {@code p}, so what is uploaded is exactly what is expected back. */
    private static double[] randomValues(GpuTestPrecision p, int n, long seed) {
        Random r = new Random(seed);
        double[] a = new double[n];
        for (int i = 0; i < n; i++) a[i] = p.round(r.nextGaussian());
        return a;
    }

    @TestFactory
    Stream<DynamicTest> floatingTransfersWithOffsets() {
        return forEachPrecision(p -> {
            GpuResource buf = p.alloc(ex, 100);
            double[] data = randomValues(p, 100, 1);
            p.upload(ex, buf, data);

            assertArrayEquals(Arrays.copyOfRange(data, 10, 20), p.download(ex, buf, 10, 10));

            double[] patch = {-1, -2, -3};
            p.upload(ex, buf, 50, patch, 0, 3);
            p.upload(ex, buf, 60, new double[]{9, 8, 7, 6}, 2, 2); // host offset

            double[] expected = data.clone();
            System.arraycopy(patch, 0, expected, 50, 3);
            expected[60] = 7;
            expected[61] = 6;
            assertArrayEquals(expected, p.download(ex, buf, 100));
            buf.release();
        });
    }

    @TestFactory
    Stream<DynamicTest> floatingSingleElementAccessors() {
        return forEachPrecision(p -> {
            GpuResource d = p.alloc(ex, 4);
            p.uploadAt(ex, d, 3, Math.PI);
            assertEquals(p.round(Math.PI), p.downloadAt(ex, d, 3)); // float narrows, double keeps everything
            d.release();
        });
    }

    @Test
    void intAndByteSingleElementAccessors() {
        GpuResource i = ex.allocateIntBuffer(4), b = ex.allocateByteBuffer(7);
        ex.uploadIntAt(i, 2, -123456);
        ex.uploadByteAt(b, 5, (byte) -7);
        assertEquals(-123456, ex.downloadIntAt(i, 2));
        assertEquals((byte) -7, ex.downloadByteAt(b, 5));
        i.release(); b.release();
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

    private static int elementsFor(GpuTestPrecision p, int megabytes) {
        return (megabytes << 20) / (p.isFloat() ? Float.BYTES : Double.BYTES);
    }

    @TestFactory
    Stream<DynamicTest> transferLargerThanInitialStagingBlockIsServed() {
        return forEachPrecision(p -> {
            int n = elementsFor(p, 24); // 24 MiB vs the 1 MiB initial staging block
            double[] data = randomValues(p, n, 4);
            GpuResource buf = p.alloc(ex, n);
            p.upload(ex, buf, data);
            assertArrayEquals(data, p.download(ex, buf, n));
            buf.release();
        });
    }

    @TestFactory
    @Tag("slow")
    Stream<DynamicTest> transferLargerThanMaxStagingIsChunked() {
        return forEachPrecision(p -> {
            int n = elementsFor(p, 72); // 72 MiB > the 64 MiB staging cap -> two chunks
            double[] data = randomValues(p, n, 5);
            GpuResource buf = p.alloc(ex, n);
            p.upload(ex, buf, data);
            assertArrayEquals(data, p.download(ex, buf, n));
            buf.release();
        });
    }

    // ------------------------------------------------------------------ copy

    @TestFactory
    Stream<DynamicTest> copyWithOffsets() {
        return forEachPrecision(p -> {
            double[] data = randomValues(p, 50, 6);
            GpuResource src = p.alloc(ex, 50), dst = p.alloc(ex, 50);
            p.upload(ex, src, data);
            p.fill(ex, dst, 0, 0.0, 50);
            p.copy(ex, src, 10, dst, 30, 15);

            double[] expected = new double[50];
            System.arraycopy(data, 10, expected, 30, 15);
            assertArrayEquals(expected, p.download(ex, dst, 50));
            src.release(); dst.release();
        });
    }

    @TestFactory
    Stream<DynamicTest> copyWithinHandlesOverlap() {
        return forEachPrecision(p -> {
            for (int[] c : new int[][]{{0, 4, 12}, {4, 0, 12}, {0, 20, 10}, {7, 7, 5}}) {
                double[] data = randomValues(p, 30, 7);
                GpuResource buf = p.alloc(ex, 30);
                p.upload(ex, buf, data);
                p.copyWithin(ex, buf, c[0], c[1], c[2]);

                double[] expected = data.clone();
                System.arraycopy(expected, c[0], expected, c[1], c[2]);
                assertArrayEquals(expected, p.download(ex, buf, 30), Arrays.toString(c));
                buf.release();
            }
        });
    }

    // ------------------------------------------------------------------ fills

    @TestFactory
    Stream<DynamicTest> floatingFills() {
        return forEachPrecision(p -> {
            GpuResource buf = p.alloc(ex, 20);

            p.fill(ex, buf, 0, 0.0, 20); // zero: single 32-bit pattern path
            assertArrayEquals(new double[20], p.download(ex, buf, 20));

            p.fill(ex, buf, 4, 3.25, 10); // arbitrary value with an offset
            double[] expected = new double[20];
            Arrays.fill(expected, 4, 14, 3.25);
            assertArrayEquals(expected, p.download(ex, buf, 20));
            buf.release();
        });
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
            GpuResource src = ex.allocateIntBuffer(1000), dst = ex.allocateIntBuffer(1000);
            ex.fillIntBuffer(src, 1000 + round, 1000);
            ex.copyIntBuffer(src, dst, 1000);
            src.release();

            assertEquals(1000 + round, ex.downloadIntAt(dst, 999));
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

    @TestFactory
    Stream<DynamicTest> scalarPushConstantsAreAlignedInEitherOrder() {
        return forEachPrecision(p -> {
            int n = 1000;
            double[] in = randomValues(p, n, 8);
            GpuResource inBuf = p.alloc(ex, n), outBuf = p.alloc(ex, n);
            p.upload(ex, inBuf, in);
            double[] expected = new double[n];
            for (int i = 0; i < n; i++) expected[i] = in[i] * 2.5 + 7;

            PortableKernel intFirst = scalarKernel(p, true);
            intFirst.bind(ex);
            intFirst.useBlocking(n, inBuf, outBuf, ex.scalar(7), p.scalar(ex, 2.5));
            assertArrayEquals(expected, p.download(ex, outBuf, n), p.tolerance(1.0));

            p.fill(ex, outBuf, 0, 0.0, n);
            PortableKernel valueFirst = scalarKernel(p, false);
            valueFirst.bind(ex);
            valueFirst.useBlocking(n, inBuf, outBuf, p.scalar(ex, 2.5), ex.scalar(7));
            assertArrayEquals(expected, p.download(ex, outBuf, n), p.tolerance(1.0));
            inBuf.release(); outBuf.release();
        });
    }

    @Test
    void sizeNotMultipleOfGroupDoesNotWritePastTheEnd() {
        // 1000 isn't a multiple of the default group size; the guard must stop the extra invocations
        GpuTestPrecision p = GpuTestPrecision.FLOAT;
        int n = 1000;
        GpuResource in = p.alloc(ex, n), out = p.alloc(ex, n + 64);
        p.fill(ex, in, 0, 0.0, n);
        p.fill(ex, out, 0, 99.0, n + 64);
        PortableKernel k = scalarKernel(p, true);
        k.bind(ex);
        k.useBlocking(n, in, out, ex.scalar(1), p.scalar(ex, 1.0));

        double[] back = p.download(ex, out, n + 64);
        for (int i = 0; i < n; i++) assertEquals(1.0, back[i], "element " + i);
        for (int i = n; i < n + 64; i++) assertEquals(99.0, back[i], "guard failed at " + i);
        in.release(); out.release();
    }

    @Test
    void manyUnawaitedDispatchesExhaustDescriptorPoolGracefully() {
        // far more dispatches than the 4096-set descriptor pool, none waited on individually
        GpuResource buf = ex.allocateFloatBuffer(1);
        ex.fillFloatBuffer(buf, 0.0f, 1);
        INCREMENT.bind(ex);
        final int count = 6000;
        for (int i = 0; i < count; i++)
            INCREMENT.use(1, buf);
        assertEquals(count, ex.downloadFloatAt(buf, 0)); // also proves they ran in order, one after another
        buf.release();
    }

    @Test
    void invalidLaunchesAreRejected() {
        GpuResource buf = ex.allocateFloatBuffer(64);
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
        GpuResource buf = ex.allocateFloatBuffer(1);
        ex.fillFloatBuffer(buf, 5.0f, 1);
        INCREMENT.bind(ex);
        INCREMENT.use(0, buf);
        assertEquals(5.0f, ex.downloadFloatAt(buf, 0));
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

    // ------------------------------------------------------------------ ordering without blocking the host
    //
    // Every operation is submitted without waiting; a buffer remembers its last operation and only work that touches
    // it is ordered after that. These tests make the host race the GPU on purpose: each sequence below is wrong if a
    // dependency (read-after-write, write-after-read, write-after-write) is missed, and right otherwise.

    @Test
    void uploadCopiesHostDataBeforeReturning() {
        // uploads are asynchronous, so the host array must already have been copied out when the call returns
        int n = 1 << 20;
        float[] data = new float[n];
        for (int i = 0; i < n; i++) data[i] = i * 0.5f;
        GpuResource buf = ex.allocateFloatBuffer(n);

        float[] host = data.clone();
        ex.uploadFloats(buf, host, n);
        Arrays.fill(host, Float.NaN); // the caller is free to reuse its array immediately

        float[] back = new float[n];
        ex.downloadFloats(buf, back, n);
        assertArrayEquals(data, back);
        buf.release();
    }

    @Test
    void dependentWorkIsOrderedAfterAnUpload() {
        // read-after-write: the increments must see the uploaded values, not run before the upload lands
        INCREMENT.bind(ex);
        final int n = 1 << 16;
        GpuResource buf = ex.allocateFloatBuffer(n);
        for (int round = 0; round < 50; round++) {
            float[] data = new float[n];
            Arrays.fill(data, round);
            ex.uploadFloats(buf, data, n); // nothing waits on this
            INCREMENT.use(n, buf);
            INCREMENT.use(n, buf);
            INCREMENT.use(n, buf);

            float[] back = new float[n];
            ex.downloadFloats(buf, back, n);
            for (int i = 0; i < n; i += 4099)
                assertEquals(round + 3f, back[i], "round " + round + " element " + i);
            assertEquals(round + 3f, back[n - 1], "round " + round + " last element");
        }
        buf.release();
    }

    @Test
    void writeAfterReadIsOrdered() {
        // the second fill overwrites the buffer the copy is still reading: the copy must see the OLD contents
        final int n = 1 << 16;
        GpuResource a = ex.allocateIntBuffer(n), b = ex.allocateIntBuffer(n);
        for (int round = 0; round < 200; round++) {
            ex.fillIntBuffer(a, 2 * round, n);
            ex.copyIntBuffer(a, b, n);
            ex.fillIntBuffer(a, 2 * round + 1, n);

            assertEquals(2 * round, ex.downloadIntAt(b, n - 1), "copy saw the later write, round " + round);
            assertEquals(2 * round + 1, ex.downloadIntAt(a, n - 1), "later write was lost, round " + round);
        }
        a.release(); b.release();
    }

    @Test
    void writesToOneBufferLandInProgramOrder() {
        // write-after-write over overlapping ranges: the host-side replay below defines the expected result
        final int n = 1 << 14;
        GpuResource buf = ex.allocateFloatBuffer(n);
        for (int round = 0; round < 50; round++) {
            float[] expected = new float[n];

            float[] first = new float[n];
            Arrays.fill(first, 1f + round);
            ex.uploadFloats(buf, first, n);
            System.arraycopy(first, 0, expected, 0, n);

            ex.fillFloatBuffer(buf, 0, 7f, n / 2);
            Arrays.fill(expected, 0, n / 2, 7f);

            float[] second = new float[n / 2];
            Arrays.fill(second, -3f - round);
            ex.uploadFloats(buf, n / 4, second, 0, n / 2);
            System.arraycopy(second, 0, expected, n / 4, n / 2);

            float[] back = new float[n];
            ex.downloadFloats(buf, back, n);
            assertArrayEquals(expected, back, "round " + round);
        }
        buf.release();
    }

    @Test
    void independentBuffersDoNotInterfere() {
        // two interleaved dependent chains on different buffers: neither may corrupt or skip the other
        INCREMENT.bind(ex);
        final int n = 1024, steps = 500;
        GpuResource a = ex.allocateFloatBuffer(n), b = ex.allocateFloatBuffer(n);
        ex.fillFloatBuffer(a, 0f, n);
        ex.fillFloatBuffer(b, 1000f, n);
        for (int i = 0; i < steps; i++) {
            INCREMENT.use(n, a);
            INCREMENT.use(n, b);
        }
        float[] backA = new float[n], backB = new float[n];
        ex.downloadFloats(a, backA, n);
        ex.downloadFloats(b, backB, n);
        for (int i = 0; i < n; i++) {
            assertEquals((float) steps, backA[i], "a[" + i + "]");
            assertEquals(1000f + steps, backB[i], "b[" + i + "]");
        }
        a.release(); b.release();
    }

    @Test
    void pendingEventsAreTrackedPerResourceAndClearOnCompletion() {
        INCREMENT.bind(ex);
        final int n = 1 << 12;
        GpuResource busy = ex.allocateFloatBuffer(n), untouched = ex.allocateFloatBuffer(n);
        assertEquals(0, untouched.pendingEvents().length, "a fresh buffer has nothing to wait for");

        ex.fillFloatBuffer(busy, 0f, n);
        for (int i = 0; i < 2000; i++)
            INCREMENT.use(n, busy);

        // the GPU may or may not be done by now; if it isn't, the event must be a real, positive serial
        for (ResourceEvent e : busy.pendingEvents()) {
            VulkanEvent event = assertInstanceOf(VulkanEvent.class, e);
            assertTrue(event.serial() > 0);
        }
        assertEquals(0, untouched.pendingEvents().length, "work on another buffer must not leak onto this one");

        ex.finish();
        assertEquals(0, busy.pendingEvents().length, "nothing is pending once the queue has drained");
        assertDoesNotThrow(busy::settle);

        // a blocking read waits for its own work, so the buffer it read from is settled afterwards
        INCREMENT.use(n, busy);
        assertEquals(2001f, ex.downloadFloatAt(busy, 0));
        assertEquals(0, busy.pendingEvents().length);
        busy.release(); untouched.release();
    }

    @Test
    void manyAsyncUploadsRecycleStagingBlocks() {
        // each upload takes a staging block that is only returned when its submission finishes; none is waited on
        final int slices = 1500, len = 256;
        GpuResource buf = ex.allocateFloatBuffer(slices * len);
        float[] expected = new float[slices * len];
        for (int s = 0; s < slices; s++) {
            float[] chunk = new float[len];
            for (int i = 0; i < len; i++) chunk[i] = s + i / 1024f;
            ex.uploadFloats(buf, s * len, chunk, 0, len);
            System.arraycopy(chunk, 0, expected, s * len, len);
        }
        float[] back = new float[slices * len];
        ex.downloadFloats(buf, back, slices * len);
        assertArrayEquals(expected, back);
        buf.release();
    }

    @Test
    void concurrentThreadsCanShareTheExecutor() throws Exception {
        INCREMENT.bind(ex);
        final int threads = 4, iterations = 40, n = 4096;
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        try {
            List<Future<?>> tasks = new ArrayList<>();
            for (int t = 0; t < threads; t++) {
                final int id = t;
                tasks.add(pool.submit(() -> {
                    GpuResource buf = ex.allocateFloatBuffer(n);
                    try {
                        for (int it = 0; it < iterations; it++) {
                            float base = id * 1000 + it;
                            float[] data = new float[n];
                            Arrays.fill(data, base);
                            ex.uploadFloats(buf, data, n);
                            INCREMENT.use(n, buf);
                            INCREMENT.use(n, buf);

                            float[] back = new float[n];
                            ex.downloadFloats(buf, back, n);
                            for (int i = 0; i < n; i += 511)
                                assertEquals(base + 2f, back[i], "thread " + id + " iteration " + it + " element " + i);
                        }
                    } finally {
                        buf.release();
                    }
                }));
            }
            for (Future<?> f : tasks)
                f.get(120, TimeUnit.SECONDS); // rethrows anything a worker hit
        } finally {
            pool.shutdownNow();
        }
    }

    // ------------------------------------------------------------------ lifecycle

    @Test
    void finishAndCloseAreIdempotent() {
        VulkanGpuExecutor own = VulkanTestSupport.open();
        assumeTrue(own != null);
        GpuResource buf = own.allocateFloatBuffer(8);
        own.fillFloatBuffer(buf, 1.0f, 8);
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
        GpuResource buf = own.allocateFloatBuffer(1);
        PortableKernel k = new PortableKernel("test_closed").glsl(INCREMENT.glslSource(), "shaderFloat32");
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