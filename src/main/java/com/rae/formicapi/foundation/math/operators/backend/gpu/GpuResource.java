package com.rae.formicapi.foundation.math.operators.backend.gpu;


/**
 * Opaque handle to a device resource (buffer). Concrete backends (OpenCL, Vulkan, ...)
 * each wrap their own native handle type in one of these -- e.g.
 * {@code OpenCLBackend}'s {@code cl_mem}, a future {@code VulkanBackend}'s
 * {@code VkBuffer}/allocation pair.
 *
 * <p>Nothing outside a backend's own package should ever unwrap one of
 * these. {@code Gpu*Vector} and {@code GpuExecutor} only ever hold and pass
 * around {@code GpuBuffer} references -- this is exactly what lets a buffer
 * allocated for one backend never accidentally get handed to another.
 * <p>
 * E is for blocking the resources. cl_event for OpenCL and VkEvent for Vulkan
 */
public interface GpuResource {

    /**
     * when accessing a resources and using it, we first need to make sure everything has finished
     */
    void waitForEventsAndClean();

    /**
     * add a new event to the resources when doing work on it.
     * @param newEvent event linked to a work
     */
    void add(ResourceEvent newEvent);


    void release();

}
