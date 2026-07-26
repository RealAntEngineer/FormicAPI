package com.rae.formicapi.foundation.math.operators.backend.cpu;

import java.util.concurrent.atomic.AtomicInteger;

public final class CpuExecutor {

    private final int threads;
    private final Thread[] workers;

    private volatile IntRangeTask currentTask;
    private volatile DoubleRangeTask reduceTask;
    private volatile int currentSize;
    private volatile boolean reductionMode;

    private final AtomicInteger generation = new AtomicInteger();

    private static final int STRIDE = 16;
    private final int[] completedGeneration;

    private final double[] doubleResults;

    public CpuExecutor(int threads) {
        if (threads < 1)
            throw new IllegalArgumentException();

        this.threads = threads;
        this.workers = new Thread[threads];
        this.completedGeneration = new int[threads * STRIDE];
        this.doubleResults = new double[threads];

        for (int i = 0; i < threads; i++) {
            final int id = i;
            workers[i] = new Thread(() -> workerLoop(id), "CpuWorker-" + id);
            workers[i].start();
        }
    }

    private void workerLoop(int id) {
        int seen = 0;

        while (true) {
            int gen;
            while ((gen = generation.get()) == seen)
                Thread.onSpinWait();

            seen = gen;

            int chunk = (currentSize + threads - 1) / threads;
            int start = id * chunk;
            int end = Math.min(currentSize, start + chunk);

            if (start < end) {
                if (reductionMode)
                    doubleResults[id] = reduceTask.run(start, end);
                else
                    currentTask.run(start, end);
            }

            completedGeneration[id * STRIDE] = seen;
        }
    }

    public void parallelFor(int size, IntRangeTask task) {
        if (size <= 0)
            return;

        currentTask = task;
        currentSize = size;
        reductionMode = false;

        waitForWorkers();
    }

    public double parallelReduceDouble(int size, DoubleRangeTask task) {
        if (size <= 0)
            return 0.0;

        reduceTask = task;
        currentSize = size;
        reductionMode = true;

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

    @FunctionalInterface
    public interface IntRangeTask {
        void run(int start, int end);
    }

    @FunctionalInterface
    public interface DoubleRangeTask {
        double run(int start, int end);
    }
}