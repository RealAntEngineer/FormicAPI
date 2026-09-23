package com.rae.formicapi.foundation.math.operators.backend.gpu;

/**
 * Opaque handle to a device resource (a real buffer) or an immediate/local
 * kernel argument. Concrete backends (OpenCL, Vulkan, ...) each wrap their
 * own native handle type in one of these.
 *
 * <p>Every operation touching a resource-backed instance should wait on
 * {@link #pendingEvents()} -- passed to the backend's own enqueue call as
 * a device-side wait list, never via a blocking host-side wait -- before
 * it runs, then record its own resulting event via {@link #recordEvent}
 * once enqueued (or call {@link #settle()} if the operation genuinely
 * blocked until done, e.g. a host transfer). This is what lets two
 * operations on independent resources proceed without a host-side wait
 * between them, while operations sharing a resource stay correctly
 * ordered through the event graph itself.
 *
 * <p>An immediate scalar or {@code __local} argument has no dependency
 * state at all -- {@link #pendingEvents()} is always empty for one, and
 * {@link #recordEvent}/{@link #settle} are no-ops.
 */
public interface GpuResource {

    /** Events representing operations still outstanding against this resource. Empty if there's nothing to depend on. */
    ResourceEvent[] pendingEvents();

    /**
     * Records {@code newEvent} as the (sole) outstanding operation against
     * this resource, replacing whatever was pending before -- valid
     * because {@code newEvent}'s own wait list already covered every prior
     * pending event, so waiting on it alone transitively covers them all.
     */
    void recordEvent(ResourceEvent newEvent);

    /** Marks this resource fully settled: nothing outstanding. Only call after an operation known to have actually completed (e.g. a blocking host transfer). */
    void settle();

    void release();
}