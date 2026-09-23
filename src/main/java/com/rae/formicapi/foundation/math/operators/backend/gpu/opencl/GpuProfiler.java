
package com.rae.formicapi.foundation.math.operators.backend.gpu.opencl;

import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_event;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.jocl.CL.*;

/**
 * Collects OpenCL event-profiling timestamps for every labeled operation
 * dispatched while attached to an {@link OpenCLGpuExecutor} (via
 * {@link OpenCLGpuExecutor#setProfiler}), aggregated per label once
 * {@link #flush()} is called.
 *
 * <p>{@link OpenCLGpuExecutor} always creates its queue with
 * {@code CL_QUEUE_PROFILING_ENABLE}, so profiling data is available
 * regardless of whether a profiler happens to be attached -- attaching one
 * just means {@link OpenCLKernel#use}/{@code useBlocking} and the
 * async buffer ops (fill/copy) additionally retain their event and hand it
 * here instead of it being purely for dependency tracking.
 *
 * <p><b>Usage:</b> attach a profiler, run the work you want measured, call
 * {@code executor.finish()} so every recorded event is guaranteed
 * complete, then call {@link #flush()}. Reading {@code CL_PROFILING_*}
 * info off an event before it's complete is invalid, so this defers
 * reading until you explicitly ask for it rather than trying to read
 * eagerly off some background thread.
 */
public final class GpuProfiler {

    private record Recorded(String label, cl_event event) {
    }

    private final List<Recorded> pending = new ArrayList<>();

    /** Called by OpenCLKernel/OpenCLGpuExecutor for every dispatched op while this profiler is attached. Retains its own reference to {@code event}, released again in {@link #flush()}. */
    synchronized void record(String label, cl_event event) {
        clRetainEvent(event);
        pending.add(new Recorded(label, event));
    }

    /**
     * Reads profiling timestamps off every event recorded since the last
     * flush, releases them, and returns per-label aggregate stats. Every
     * recorded event must already be complete (call {@code executor.finish()}
     * first) -- profiling queries are only valid on a completed event.
     */
    public synchronized Map<String, Stats> flush() {
        Map<String, Stats> byLabel = new LinkedHashMap<>();

        for (Recorded r : pending) {
            long queued   = profilingInfo(r.event(), CL_PROFILING_COMMAND_QUEUED);
            long start    = profilingInfo(r.event(), CL_PROFILING_COMMAND_START);
            long end      = profilingInfo(r.event(), CL_PROFILING_COMMAND_END);

            byLabel.computeIfAbsent(r.label(), Stats::new).add(queued, start, end);

            clReleaseEvent(r.event());
        }

        pending.clear();
        return byLabel;
    }

    private static long profilingInfo(cl_event event, int param) {
        long[] out = new long[1];
        clGetEventProfilingInfo(event, param, Sizeof.cl_ulong, Pointer.to(out), null);
        return out[0];
    }

    /**
     * Per-label aggregate over every call recorded under that label:
     * count, actual device execution time ({@code end - start}), and the
     * queue-to-start latency ({@code start - queued}) -- time spent
     * waiting on dependencies/queue backlog before the device actually
     * began, as distinct from time spent doing the work.
     */
    public static final class Stats {
        public final String label;
        public long count;
        public long totalExecNanos;
        public long totalWaitNanos;
        public long minExecNanos = Long.MAX_VALUE;
        public long maxExecNanos = Long.MIN_VALUE;

        private Stats(String label) {
            this.label = label;
        }

        private void add(long queued, long start, long end) {
            long exec = end - start;
            long wait = start - queued;
            count++;
            totalExecNanos += exec;
            totalWaitNanos += wait;
            if (exec < minExecNanos) minExecNanos = exec;
            if (exec > maxExecNanos) maxExecNanos = exec;
        }

        public double avgExecNanos() {
            return count == 0 ? 0 : (double) totalExecNanos / count;
        }

        public double avgWaitNanos() {
            return count == 0 ? 0 : (double) totalWaitNanos / count;
        }

        @Override
        public String toString() {
            return String.format(
                    "%-24s count=%-8d totalExec=%10.3f ms  avgExec=%9.1f ns  minExec=%8d ns  maxExec=%8d ns  avgQueueWait=%9.1f ns",
                    label, count, totalExecNanos / 1e6, avgExecNanos(),
                    minExecNanos == Long.MAX_VALUE ? 0 : minExecNanos,
                    maxExecNanos == Long.MIN_VALUE ? 0 : maxExecNanos,
                    avgWaitNanos());
        }
    }
}