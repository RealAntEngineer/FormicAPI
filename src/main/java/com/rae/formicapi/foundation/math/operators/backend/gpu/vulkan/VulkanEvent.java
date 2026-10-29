package com.rae.formicapi.foundation.math.operators.backend.gpu.vulkan;

import com.rae.formicapi.foundation.math.operators.backend.gpu.ResourceEvent;

/**
 * Vulkan's {@link ResourceEvent}: the serial number of one queue submission.
 *
 * <p>Every submission made by {@link VulkanGpuExecutor} gets the next value of
 * a monotonically increasing counter and signals that value on the executor's
 * timeline semaphore when it finishes. Because the queue is in-order for
 * semaphore signalling, "the semaphore reached N" means submission N <em>and
 * everything submitted before it</em> has completed -- so, like an OpenCL
 * {@code cl_event}, holding the latest serial is enough to cover all earlier
 * ones.
 *
 * <p>Unlike a {@code cl_event} there is nothing to retain or release: a serial
 * is just a number.
 */
public record VulkanEvent(long serial) implements ResourceEvent {
}