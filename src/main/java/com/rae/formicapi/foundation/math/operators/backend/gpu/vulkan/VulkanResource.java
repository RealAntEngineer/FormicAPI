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
 * <p><b>Per-resource event tracking.</b> Like {@code OpenCLResource}, a buffer
 * remembers the last operation that touched it, as a {@link VulkanEvent} (the
 * serial of a queue submission). The executor looks at it when the buffer is
 * used again: if that submission hasn't finished yet it opens the new
 * submission with a barrier (a device-side wait -- the host never blocks);
 * if it has finished, or the buffer was never used, nothing is inserted and
 * the new work is free to overlap with whatever else is in flight. Only the
 * <em>latest</em> serial is kept: a later submission's barrier already covers
 * every earlier one, so waiting on the newest transitively covers them all
 * (the same argument {@link GpuResource#recordEvent} makes for OpenCL).
 *
 * <p>{@link #release()} defers destruction until the buffer's own last
 * operation has finished, not until the whole queue has drained.
 */
public final class VulkanResource implements GpuResource {

    private static final ResourceEvent[] NONE = new ResourceEvent[0];

    final Kind kind;

    // BUFFER
    final long buffer;
    final long memory;
    final long sizeBytes;
    private final @Nullable VulkanGpuExecutor owner;
    private boolean released;
    /** Serial of the last submission that touched this buffer; 0 = nothing outstanding. */
    private long pendingSerial;

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

    /** Raw accessor for the executor: serial of the last operation that touched this buffer, 0 if none. May be stale (already finished). */
    synchronized long pendingSerial() {
        return pendingSerial;
    }

    @Override
    public synchronized ResourceEvent[] pendingEvents() {
        if (kind != Kind.BUFFER || pendingSerial == 0)
            return NONE;
        assert owner != null;
        if (owner.isComplete(pendingSerial)) { // finished since we last looked: drop it lazily
            pendingSerial = 0;
            return NONE;
        }
        return new ResourceEvent[]{new VulkanEvent(pendingSerial)};//I don't agree with that.
    }

    @Override
    public synchronized void recordEvent(ResourceEvent newEvent) {
        if (kind != Kind.BUFFER) return; // nothing to depend on for an immediate/local arg
        if (!(newEvent instanceof VulkanEvent(long serial)))
            throw new IllegalArgumentException("Not a Vulkan event: " + newEvent.getClass());
        if (serial > pendingSerial) // serials only grow; never let a late-arriving older one overwrite a newer one
            pendingSerial = serial;
    }

    /** Marks this resource fully settled. Only call after an operation known to have actually completed. */
    @Override
    public synchronized void settle() {
        pendingSerial = 0;
    }

    /**
     * Settles the resource only if nothing newer than {@code serial} was
     * recorded meanwhile -- what a host-side wait on {@code serial} (done
     * outside the executor lock, so another thread may have enqueued more work
     * on this buffer in the meantime) is entitled to conclude.
     */
    synchronized void settleThrough(long serial) {
        if (pendingSerial <= serial)
            pendingSerial = 0;
    }

    @Override
    public void release() {
        long serial;
        synchronized (this) { // don't hold our monitor while calling into the executor (it takes its own lock, then ours)
            if (kind != Kind.BUFFER || released)
                return;
            released = true;
            serial = pendingSerial;
        }
        assert owner != null;
        owner.retire(buffer, memory, serial);
    }
}