package com.rae.formicapi.foundation.math.operators.backend.gpu.vulkan;

import com.rae.formicapi.foundation.math.operators.backend.gpu.GpuResource;
import com.rae.formicapi.foundation.math.operators.backend.gpu.ResourceEvent;
import org.jetbrains.annotations.Nullable;

/**
 * One Vulkan kernel argument / device resource -- the Vulkan counterpart of
 * {@code OpenCLResource}, with the same three kinds sharing one type so
 * {@code Kernel#use} can take a single flat argument list:
 * <ul>
 *   <li>{@link Kind#BUFFER}: a real {@code VkBuffer} plus its
 *       {@code VkDeviceMemory}, created by {@link VulkanGpuExecutor};</li>
 *   <li>{@link Kind#IMMEDIATE}: an immediate {@code int}/{@code double}/{@code float}, delivered to the
 *       shader through the push-constant block;</li>
 *   <li>{@link Kind#LOCAL}: work-group scratch. Vulkan declares that inside the
 *       shader ({@code shared} arrays), so this kind carries no data and is
 *       skipped at dispatch -- it only exists so call sites stay identical
 *       across backends.</li>
 * </ul>
 *
 * <p><b>No per-resource event graph.</b> OpenCL tracks outstanding events per
 * buffer; here every submission starts with a global memory barrier on the
 * single compute queue, which already orders everything in submission order,
 * so {@link #pendingEvents()} is always empty and {@link #recordEvent} /
 * {@link #settle()} are no-ops. {@link #release()} defers the actual
 * destruction until the GPU is done with every command submitted so far.
 */
public final class VulkanResource implements GpuResource {

    private static final ResourceEvent[] NONE = new ResourceEvent[0];
    //TODO Create the events for Vulkan, Vulkan supports it and that's way cleaner than walling it

    final Kind kind;

    // BUFFER
    final long buffer;
    final long memory;
    final long sizeBytes;
    private final @Nullable VulkanGpuExecutor owner;
    private boolean released;

    // SCALAR
    final long bits;//Don't do that, send everything to a buffer a scalar is a buffer of size 1.
    final int  width; // 4 or 8 bytes

    private VulkanResource(Kind kind, long buffer, long memory, long sizeBytes, @Nullable VulkanGpuExecutor owner, long bits, int width) {
        this.kind = kind;
        this.buffer = buffer;
        this.memory = memory;
        this.sizeBytes = sizeBytes;
        this.owner = owner;
        this.bits = bits;
        this.width = width;
    }

    static VulkanResource buffer(VulkanGpuExecutor owner, long buffer, long memory, long sizeBytes) {
        return new VulkanResource(Kind.BUFFER, buffer, memory, sizeBytes, owner, 0, 0);
    }

    static VulkanResource of(int value) {
        return new VulkanResource(Kind.IMMEDIATE, 0, 0, 0, null, value, Integer.BYTES);
    }

    static VulkanResource of(double value) {
        return new VulkanResource(Kind.IMMEDIATE, 0, 0, 0, null, Double.doubleToRawLongBits(value), Double.BYTES);
    }

    static VulkanResource of(float value) {
        return new VulkanResource(Kind.IMMEDIATE, 0, 0, 0, null, Float.floatToRawIntBits(value), Float.BYTES);
    }

    static VulkanResource local(long bytes) {
        return new VulkanResource(Kind.LOCAL, 0, 0, bytes, null, 0, 0);
    }

    boolean isBuffer() {
        return kind == Kind.BUFFER;
    }

    @Override
    public ResourceEvent[] pendingEvents() {
        return NONE;
    }

    @Override
    public void recordEvent(ResourceEvent newEvent) {
        // ordering is enforced by the executor's per-submission global barrier, see class doc
    }

    @Override
    public void settle() {

    }

    @Override
    public synchronized void release() {
        if (kind != Kind.BUFFER || released)
            return;
        released = true;
        assert owner != null;
        owner.retire(buffer, memory);
    }
}
