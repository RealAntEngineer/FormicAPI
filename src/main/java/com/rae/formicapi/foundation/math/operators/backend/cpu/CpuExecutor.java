package com.rae.formicapi.foundation.math.operators.backend.cpu;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.LinkedBlockingQueue;

public final class CpuExecutor {

    private final int threads;
    private final Thread[] workers;

    private final BlockingQueue<Runnable> queue =
            new LinkedBlockingQueue<>();

    public CpuExecutor(int threads) {

        if (threads < 1)
            throw new IllegalArgumentException();

        this.threads = threads;
        this.workers = new Thread[threads];

        for (int i = 0; i < threads; i++) {
            workers[i] = new Thread(this::workerLoop);
            workers[i].start();
        }
    }


    private void workerLoop() {
        try {
            while (true) {
                Runnable task = queue.take();
                task.run();
            }
        }
        catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }


    public void submit(Runnable task) {
        queue.add(task);
    }

    public void parallelFor(int size, IntRangeTask task) {

        int chunk = (size + threads - 1) / threads;

        CountDownLatch latch = new CountDownLatch(threads);

        for (int i = 0; i < threads; i++) {

            int start = i * chunk;
            int end = Math.min(size, start + chunk);

            submit(() -> {
                try {
                    if (start < end)
                        task.run(start, end);
                }
                finally {
                    latch.countDown();
                }
            });
        }

        try {
            long start = System.nanoTime();
            latch.await();
            System.out.println("latch took "+(System.nanoTime() - start));
        }
        catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Parallel operation interrupted", e);
        }
    }

    @FunctionalInterface
    public interface IntRangeTask {
        void run(int start, int end);
    }
}