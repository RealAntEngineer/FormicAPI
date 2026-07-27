package com.rae.formicapi.foundation.math.operators.backend.cpu;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;

public final class CpuExecutor implements AutoCloseable {

    /**
     * Default cutoff below which callers (CpuVector, CpuPaddedCSRMatrix)
     * should just run serially instead of paying dispatch overhead. This is
     * a rough default, not a measured constant - the real crossover point
     * depends on core count, memory bandwidth and per-element cost, and can
     * vary by 10x+ across machines. Use {@link #calibrateThreshold(int)} to
     * measure a real value for the target hardware, then either pass it to
     * the constructor or set it with {@link #setParallelThreshold(int)}.
     */
    public static final int DEFAULT_PARALLEL_THRESHOLD = 4096;
    private static final int   STRIDE = 16;
    // Candidate sizes to sweep when calibrating, smallest to largest.
    private static final int[] DEFAULT_CANDIDATE_SIZES = {
            256, 512, 1_024, 2_048, 4_096, 8_192, 16_384,
            32_768, 65_536, 131_072, 262_144, 524_288, 1_048_576
    };
    private static final int WARMUP_TRIALS = 3;
    private static final int TIMED_TRIALS  = 7;
    // Prevents the JIT from optimizing away the benchmarked work as dead code.
    private static volatile double blackhole;
    private final int               threads;
    private final CpuWorkerThread[] workers;
    private final AtomicInteger generation = new AtomicInteger();
    private final        int[] completedGeneration;
    private final double[] doubleResults;
    // Per-thread scratch buffers for parallelForAccumulate, sized lazily and
    // reused across calls (resized only when the output length changes).
    // Cleared fully at the start of every accumulate call - O(outputSize)
    // per thread, unconditionally, regardless of how many entries actually
    // get written. Each thread only ever touches its own buffer during the
    // parallel phase - no shared memory writes, so no cross-thread
    // contention at all, atomic or otherwise. The tradeoff is the fixed
    // O(outputSize) clear + reduce cost per call, which is wasteful when
    // outputSize is much larger than the number of actual writes per call.
    private final double[][] accumBuffers;
    private volatile IntRangeTask    currentTask;
    private volatile DoubleRangeTask reduceTask;
    private volatile AccumulateTask  accumulateTask;
    private volatile int             currentSize;
    private volatile Mode            mode;
    private       int        accumOutputSize = -1;
    private volatile int     parallelThreshold;
    private volatile boolean shutdown = false;

    public CpuExecutor(int threads) {
        this(threads, DEFAULT_PARALLEL_THRESHOLD);
    }

    public CpuExecutor(int threads, int parallelThreshold) {
        if (threads < 1)
            throw new IllegalArgumentException();
        if (parallelThreshold < 0)
            throw new IllegalArgumentException();

        this.threads = threads;
        this.parallelThreshold = parallelThreshold;
        this.workers = new CpuWorkerThread[threads];
        this.completedGeneration = new int[threads * STRIDE];
        this.doubleResults = new double[threads];
        this.accumBuffers = new double[threads][];

        for (int i = 0; i < threads; i++) {
            workers[i] = new CpuWorkerThread(this, i);
            workers[i].setDaemon(true);
            workers[i].start();
        }
    }

    /**
     * Measures, on this machine, the smallest array size at which dispatching
     * work across {@code threads} workers actually beats running it serially,
     * and returns that as a threshold suitable for
     * {@link #setParallelThreshold(int)}.
     *
     * <p>Spins up its own short-lived {@code CpuExecutor} (shut down before
     * returning) and times an axpy-like elementwise workload
     * ({@code d[i] += a * x[i]}) — representative of the per-element cost of
     * most {@code CpuVector} operations — at a sweep of sizes, comparing a
     * plain serial loop against {@link #parallelFor}. For each size it takes
     * the median of several timed trials (after a few discarded warmup
     * trials to let the JIT settle) and returns the first size where
     * parallel is faster.
     *
     * <p>This is a coarse, single-workload estimate, not a precision
     * benchmark — treat the result as a reasonable starting point, not an
     * exact number. It also takes real wall-clock time to run (a handful of
     * seconds), so call it once at startup / during tuning, not per-request.
     *
     * @param threads number of worker threads to calibrate for
     * @return recommended {@code parallelThreshold}; the largest candidate
     * size if parallel dispatch never wins in the sweep
     */
    public static int calibrateThreshold(int threads) {
        return calibrateThreshold(threads, DEFAULT_CANDIDATE_SIZES);
    }

    public static int calibrateThreshold(int threads, int[] candidateSizes) {
        if (candidateSizes.length == 0)
            throw new IllegalArgumentException("need at least one candidate size");

        CpuExecutor exec = new CpuExecutor(threads, 0);
        try {
            for (int size : candidateSizes) {
                double[] a = new double[size];
                double[] x = new double[size];
                for (int i = 0; i < size; i++) {
                    a[i] = i * 0.5 + 1.0;
                    x[i] = i * 0.25 - 1.0;
                }

                long serialTime = medianNanos(() -> {
                    double sum = 0.0;
                    for (int i = 0; i < size; i++) {
                        a[i] += 1.0000001 * x[i];
                        sum += a[i];
                    }
                    blackhole = sum;
                });

                long parallelTime = medianNanos(() -> {
                    exec.parallelFor(size, (start, end) -> {
                        double sum = 0.0;
                        for (int i = start; i < end; i++) {
                            a[i] += 1.0000001 * x[i];
                            sum += a[i];
                        }
                        blackhole = sum;
                    });
                });

                if (parallelTime < serialTime)
                    return size;
            }

            return candidateSizes[candidateSizes.length - 1];
        } finally {
            exec.shutdown();
        }
    }

    private static long medianNanos(Runnable work) {
        for (int i = 0; i < WARMUP_TRIALS; i++)
            work.run();

        long[] samples = new long[TIMED_TRIALS];
        for (int i = 0; i < TIMED_TRIALS; i++) {
            long start = System.nanoTime();
            work.run();
            samples[i] = System.nanoTime() - start;
        }

        Arrays.sort(samples);
        return samples[samples.length / 2];
    }

    public int getThreadCount() {
        return threads;
    }

    /**
     * Work sizes at or above this should be dispatched in parallel; below it, run serially.
     */
    public int getParallelThreshold() {
        return parallelThreshold;
    }

    public void setParallelThreshold(int parallelThreshold) {
        if (parallelThreshold < 0)
            throw new IllegalArgumentException();
        this.parallelThreshold = parallelThreshold;
    }

    @Override
    public void close() {
        shutdown();
    }

    /**
     * Stops the worker threads. Safe to call once; the executor is unusable
     * afterwards. Mainly needed so short-lived executors (e.g. the ones
     * {@link #calibrateThreshold(int)} spins up internally) don't leak
     * spinning threads.
     *
     * <p><b>Every {@code CpuExecutor} must eventually be shut down.</b> Its
     * worker threads busy-spin with {@link Thread#onSpinWait()} while idle
     * rather than blocking, which means an un-shut-down executor doesn't
     * just leak memory - it permanently pins one CPU core per worker, even
     * doing nothing. In tests especially, this compounds: each test that
     * creates its own executor and forgets to shut it down leaves its
     * workers spinning for the rest of the JVM's life, so by the Nth test
     * you can have N x threads worth of cores oversubscribed, and parallel
     * work that looks like a regression is often just contention with
     * threads from an earlier test. Prefer try-with-resources
     * ({@code CpuExecutor} implements {@link AutoCloseable}) or an
     * {@code @AfterEach}/{@code finally} block over relying on GC.
     */
    public void shutdown() {
        shutdown = true;
        generation.incrementAndGet();
    }

    private void workerLoop(int id) {
        int seen = 0;

        while (true) {
            int gen;
            while ((gen = generation.get()) == seen) {
                if (shutdown)
                    return;
                Thread.onSpinWait();
            }

            seen = gen;

            if (shutdown)
                return;

            int chunk = (currentSize + threads - 1) / threads;
            int start = id * chunk;
            int end   = Math.min(currentSize, start + chunk);

            switch (mode) {
                case FOR -> {
                    if (start < end)
                        currentTask.run(start, end);
                }
                case REDUCE -> doubleResults[id] = (start < end) ? reduceTask.run(start, end) : 0.0;
                case ACCUMULATE -> {
                    // No Arrays.fill here: buffers start zero (fresh double[]
                    // arrays default to 0.0) and get zeroed again during the
                    // reduce pass below, so a separate O(cols) clear here
                    // would just be paying for the same reset twice.
                    if (start < end)
                        accumulateTask.run(start, end, accumBuffers[id]);
                }
            }

            completedGeneration[id * STRIDE] = seen;
        }
    }

    public double parallelReduceDouble(int size, DoubleRangeTask task) {
        if (size <= 0)
            return 0.0;

        reduceTask = task;
        currentSize = size;
        mode = Mode.REDUCE;

        waitForWorkers();

        double result = 0;
        for (double value : doubleResults)
            result += value;

        return result;
    }

    private void waitForWorkers() {
        int job = generation.incrementAndGet();

        while (true) {
            boolean done = true;

            for (int i = 0; i < threads; i++) {
                if (completedGeneration[i * STRIDE] != job) {
                    done = false;
                    break;
                }
            }

            if (done)
                return;

            Thread.onSpinWait();
        }
    }

    /**
     * Runs {@code task} over {@code [0, size)} in parallel, where each worker
     * accumulates into its own private buffer of length {@code output.length}
     * instead of writing directly into shared memory. This is the pattern
     * needed for scatter-add workloads (e.g. transpose-multiply of a sparse
     * matrix) where two different row ranges can legitimately touch the same
     * output index and a plain {@link #parallelFor} would race.
     *
     * <p>Each thread's buffer starts and ends every call at all-zero, and
     * gets summed into {@code output} once every worker has finished -
     * O(output.length) per call regardless of how many entries a given call
     * actually writes. There's no separate clear pass: buffers are reset to
     * zero as a side effect of being folded into {@code output} (both the
     * clear and the reduce would otherwise be O(output.length), so doing
     * them as one pass instead of two roughly halves this operation's fixed
     * overhead). {@code output} itself is added to, not overwritten -
     * callers that want a fresh result should clear it first (as
     * {@code transposeApply} does).
     *
     * <p>Deliberately not atomics-based: each thread only ever touches its
     * own private buffer during the parallel phase, so there's no shared
     * memory contention to pay for - not even the "no logical collision but
     * still contends" cost an atomic CAS approach pays when two threads
     * write to different indices that happen to share a 64-byte cache line.
     * The tradeoff is a fixed O(output.length) cost per call no matter how
     * sparse the actual writes are - and that cost is itself dispatched
     * through {@link #parallelFor} (a second barrier) rather than run on the
     * calling thread alone. It's the same order of magnitude as the actual
     * per-thread compute work, so leaving it serial would cap the overall
     * speedup by Amdahl's law regardless of how well the compute phase
     * scales.
     */
    public void parallelForAccumulate(int size, double[] output, AccumulateTask task) {
        if (size <= 0)
            return;

        ensureAccumBuffers(output.length);

        accumulateTask = task;
        currentSize = size;
        mode = Mode.ACCUMULATE;

        waitForWorkers();

        parallelFor(output.length, (start, end) -> {
            for (double[] local : accumBuffers) {
                for (int i = start; i < end; i++) {
                    output[i] += local[i];
                    local[i] = 0.0;
                }
            }
        });
    }

    private void ensureAccumBuffers(int outputSize) {
        if (accumOutputSize != outputSize) {
            for (int i = 0; i < threads; i++)
                accumBuffers[i] = new double[outputSize];
            accumOutputSize = outputSize;
        }
    }

    public void parallelFor(int size, IntRangeTask task) {
        if (size <= 0)
            return;

        currentTask = task;
        currentSize = size;
        mode = Mode.FOR;

        waitForWorkers();
    }

    private enum Mode {FOR, REDUCE, ACCUMULATE}

    @FunctionalInterface
    public interface IntRangeTask {
        void run(int start, int end);
    }

    @FunctionalInterface
    public interface DoubleRangeTask {
        double run(int start, int end);
    }

    @FunctionalInterface
    public interface AccumulateTask {
        void run(int start, int end, double[] local);
    }

    /**
     * Dedicated worker thread class instead of an anonymous {@code Thread}
     * wrapping a lambda. Profilers (async-profiler, JFR, VisualVM, thread
     * dumps) key stack frames off the runnable's class; a lambda shows up as
     * an opaque synthetic name like {@code CpuExecutor$$Lambda$12/...},
     * whereas this gives every worker a stable, readable frame
     * ({@code CpuExecutor$CpuWorkerThread.run}) plus a real per-instance
     * thread name ("CpuWorker-N") that's easy to pick out in a thread list.
     */
    private static final class CpuWorkerThread extends Thread {
        private final CpuExecutor owner;
        private final int         id;

        CpuWorkerThread(CpuExecutor owner, int id) {
            super("CpuWorker-" + id);
            this.owner = owner;
            this.id = id;
        }

        @Override
        public void run() {
            owner.workerLoop(id);
        }
    }
}