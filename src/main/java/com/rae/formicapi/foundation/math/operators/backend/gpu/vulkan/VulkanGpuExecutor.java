package com.rae.formicapi.foundation.math.operators.backend.gpu.vulkan;

import com.rae.formicapi.foundation.math.operators.backend.gpu.GpuExecutor;
import com.rae.formicapi.foundation.math.operators.backend.gpu.GpuResource;
import com.rae.formicapi.foundation.math.operators.backend.gpu.Kernel;
import com.rae.formicapi.foundation.math.operators.backend.gpu.PortableKernel;
import org.jetbrains.annotations.Nullable;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.*;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;

import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.vulkan.VK10.*;
import static org.lwjgl.vulkan.VK11.*;
import static org.lwjgl.vulkan.VK12.*;

/**
 * Vulkan's {@link GpuExecutor}: owns the instance / physical + logical device /
 * single compute queue and implements the buffer lifecycle, host transfers,
 * fills, copies and compute dispatch. Like {@code OpenCLGpuExecutor} it owns
 * no kernel source -- those live as {@code PortableKernel} constants next to
 * the {@code Gpu*Vector}/{@code Gpu*Matrix} classes that use them.
 *
 * <h3>Execution model</h3>
 * <ul>
 *   <li><b>In-order queue.</b> Every operation is its own one-time command
 *       buffer submitted to one queue, and every command buffer begins with a
 *       global memory barrier (compute+transfer writes -&gt; compute+transfer
 *       reads/writes). That reproduces an in-order OpenCL queue: operations
 *       see each other's results in submission order with no per-buffer
 *       bookkeeping. Dispatches, copies and fills return immediately; host
 *       transfers and {@code useBlocking} wait on a fence.</li>
 *   <li><b>Staged transfers.</b> Buffers live in device-local memory; uploads
 *       and downloads go through one persistently mapped host-visible staging
 *       buffer (grown on demand, chunked when a transfer exceeds
 *       {@link #MAX_STAGING_BYTES}).</li>
 *   <li><b>Deferred destruction.</b> {@link GpuResource#release()} on a buffer
 *       can't free it while queued work still uses it, so destruction is
 *       queued behind the submissions issued so far and carried out as their
 *       fences complete.</li>
 *   <li><b>Locking.</b> One queue means external synchronization: every public
 *       method locks on the same monitor.</li>
 * </ul>
 *
 * <h3>Requirements and limits</h3>
 * Vulkan 1.2 and the {@code shaderFloat64} feature (selection fails fast
 * otherwise, mirroring the OpenCL backend's fp64 requirement). 64-bit buffer
 * atomics ({@code shaderBufferInt64Atomics}, needed by
 * {@code GpuPaddedCSRMatrix.transposeApply}) and 8-bit storage (needed by
 * {@code GpuBooleanVector.gather}) are enabled when present and reported
 * through {@link #supports(Kernel)}. Each buffer is its own
 * {@code vkAllocateMemory}, so very large numbers of live buffers can hit the
 * device's {@code maxMemoryAllocationCount} (4096 guaranteed); a single buffer
 * is also bound whole, so it must fit {@code maxStorageBufferRange}.
 *
 * <p>Set the system property {@code formicapi.vulkan.validation=true} to
 * enable the Khronos validation layer when it is installed.
 */
public final class VulkanGpuExecutor implements GpuExecutor, AutoCloseable {

    private static final int  DEFAULT_LOCAL_SIZE   = 256;
    private static final long INITIAL_STAGING_BYTES = 1L << 20;
    /** Transfers larger than this are split into several blocking chunks. Multiple of 8 so a chunk never splits a double. */
    static final  long MAX_STAGING_BYTES           = 64L << 20;
    private static final int  MAX_DESCRIPTOR_SETS  = 4096;

    private static final int BUFFER_USAGE =
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    private final Object lock = new Object();

    private final VkInstance instance;
    private final VkPhysicalDevice physicalDevice;
    private final VkDevice device;
    private final VkQueue queue;
    private final int queueFamily;
    private final String deviceName;

    private final VkPhysicalDeviceMemoryProperties memoryProperties;

    // capabilities (see supportsFeature)
    private final boolean hasInt64;
    private final boolean hasInt64Atomics;
    private final boolean has8BitStorage;
    private final boolean hasInt8;

    private final int  maxInvocations;
    private final int  maxGroupSizeX;
    private final int  maxGroupCountX;
    private final int  maxGroupCountY;
    private final int  defaultLocalSize;

    private final long commandPool;
    private final long descriptorPool;

    // submission tracking
    private static final class Submission {
        final VkCommandBuffer cb;
        final long fence;
        final List<Long> descriptorSets = new ArrayList<>(2);
        long serial;

        Submission(VkCommandBuffer cb, long fence) {
            this.cb = cb;
            this.fence = fence;
        }
    }

    private record Dead(long buffer, long memory, long serial) {
    }

    private final ArrayDeque<Submission> inFlight = new ArrayDeque<>();
    private final ArrayDeque<Submission> idle = new ArrayDeque<>();
    private final List<Dead> graveyard = new ArrayList<>();
    private long submitSerial = 0;
    private long completedSerial = 0;

    // staging
    private long stagingBuffer = VK_NULL_HANDLE;
    private long stagingMemory = VK_NULL_HANDLE;
    private long stagingPtr = 0;
    private long stagingCapacity = 0;

    private final List<VulkanKernel> kernels = new ArrayList<>();
    private final VulkanKernel fillDoubleKernel;

    private volatile boolean closed = false;

    /** Opens a device on the best Vulkan 1.2 GPU with fp64 support (discrete &gt; integrated &gt; virtual &gt; CPU). */
    public VulkanGpuExecutor() {
        this(-1);
    }

    /**
     * Opens the physical device at {@code deviceIndex} in the driver's
     * enumeration order (see {@link #deviceNames()}); {@code -1} auto-selects.
     */
    public VulkanGpuExecutor(int deviceIndex) {
        try (MemoryStack stack = stackPush()) {
            this.instance = createInstance(stack);

            VkPhysicalDevice[] all = enumerate(stack, instance);
            if (all.length == 0)
                throw new RuntimeException("No Vulkan physical devices found");

            VkPhysicalDevice chosen = deviceIndex >= 0 ? requireUsable(stack, all, deviceIndex) : selectDefault(stack, all);
            this.physicalDevice = chosen;
            this.queueFamily = findComputeQueueFamily(stack, chosen);

            VkPhysicalDeviceProperties props = VkPhysicalDeviceProperties.calloc(stack);
            vkGetPhysicalDeviceProperties(chosen, props);
            this.deviceName = props.deviceNameString();
            VkPhysicalDeviceLimits limits = props.limits();
            this.maxInvocations = limits.maxComputeWorkGroupInvocations();
            this.maxGroupSizeX = limits.maxComputeWorkGroupSize(0);
            this.maxGroupCountX = limits.maxComputeWorkGroupCount(0);
            this.maxGroupCountY = limits.maxComputeWorkGroupCount(1);

            VkPhysicalDeviceVulkan12Features supported12 = VkPhysicalDeviceVulkan12Features.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES);
            VkPhysicalDeviceFeatures2 supported = VkPhysicalDeviceFeatures2.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2)
                    .pNext(supported12.address());
            vkGetPhysicalDeviceFeatures2(chosen, supported);

            if (!supported.features().shaderFloat64())
                throw new RuntimeException("Vulkan device '" + deviceName + "' doesn't support shaderFloat64 (double precision)");

            this.hasInt64 = supported.features().shaderInt64();
            this.hasInt64Atomics = hasInt64 && supported12.shaderBufferInt64Atomics();
            this.has8BitStorage = supported12.storageBuffer8BitAccess();
            this.hasInt8 = supported12.shaderInt8();

            // enable exactly what we use and what's available
            VkPhysicalDeviceVulkan12Features enable12 = VkPhysicalDeviceVulkan12Features.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES)
                    .shaderBufferInt64Atomics(hasInt64Atomics)
                    .storageBuffer8BitAccess(has8BitStorage)
                    .shaderInt8(hasInt8);
            VkPhysicalDeviceFeatures2 enable = VkPhysicalDeviceFeatures2.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2)
                    .pNext(enable12.address());
            enable.features().shaderFloat64(true).shaderInt64(hasInt64);

            VkDeviceQueueCreateInfo.Buffer queueInfo = VkDeviceQueueCreateInfo.calloc(1, stack);
            queueInfo.get(0)
                    .sType(VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO)
                    .queueFamilyIndex(queueFamily)
                    .pQueuePriorities(stack.floats(1.0f));

            VkDeviceCreateInfo deviceInfo = VkDeviceCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO)
                    .pNext(enable.address())
                    .pQueueCreateInfos(queueInfo);

            PointerBuffer pDevice = stack.mallocPointer(1);
            check(vkCreateDevice(chosen, deviceInfo, null, pDevice), "vkCreateDevice");
            this.device = new VkDevice(pDevice.get(0), chosen, deviceInfo);

            PointerBuffer pQueue = stack.mallocPointer(1);
            vkGetDeviceQueue(device, queueFamily, 0, pQueue);
            this.queue = new VkQueue(pQueue.get(0), device);

            this.memoryProperties = VkPhysicalDeviceMemoryProperties.calloc();
            vkGetPhysicalDeviceMemoryProperties(chosen, memoryProperties);

            VkCommandPoolCreateInfo poolInfo = VkCommandPoolCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO)
                    .flags(VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT)
                    .queueFamilyIndex(queueFamily);
            LongBuffer pPool = stack.mallocLong(1);
            check(vkCreateCommandPool(device, poolInfo, null, pPool), "vkCreateCommandPool");
            this.commandPool = pPool.get(0);

            VkDescriptorPoolSize.Buffer poolSizes = VkDescriptorPoolSize.calloc(1, stack);
            poolSizes.get(0)
                    .type(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .descriptorCount(MAX_DESCRIPTOR_SETS * VulkanKernel.MAX_BINDINGS);
            VkDescriptorPoolCreateInfo dpInfo = VkDescriptorPoolCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO)
                    .flags(VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT)
                    .maxSets(MAX_DESCRIPTOR_SETS)
                    .pPoolSizes(poolSizes);
            LongBuffer pDescPool = stack.mallocLong(1);
            check(vkCreateDescriptorPool(device, dpInfo, null, pDescPool), "vkCreateDescriptorPool");
            this.descriptorPool = pDescPool.get(0);

            long limit = Math.min(DEFAULT_LOCAL_SIZE, Math.min(maxInvocations, maxGroupSizeX));
            int pow2 = 1;
            while (pow2 * 2L <= limit) pow2 *= 2;
            this.defaultLocalSize = pow2;
        }

        // internal helper kernel: a 64-bit fill pattern can't be expressed with vkCmdFillBuffer (32-bit only)
        this.fillDoubleKernel = registerKernel("fill_double", """
                layout(std430, binding = 0) buffer D { double d[]; };
                layout(push_constant) uniform PC { int gsz; int offset; double value; } pc;
                void main() {
                    uint i = GID;
                    if (i >= uint(pc.gsz)) return;
                    d[uint(pc.offset) + i] = pc.value;
                }
                """, "shaderFloat64");
    }

    /** Human-readable name of the selected physical device. */
    public String deviceName() {
        return deviceName;
    }

    /** Names of every Vulkan physical device the driver exposes, in the order {@link #VulkanGpuExecutor(int)} indexes them. */
    public static List<String> deviceNames() {
        try (MemoryStack stack = stackPush()) {
            VkInstance inst = createInstance(stack);
            try {
                List<String> names = new ArrayList<>();
                for (VkPhysicalDevice pd : enumerate(stack, inst)) {
                    VkPhysicalDeviceProperties props = VkPhysicalDeviceProperties.calloc(stack);
                    vkGetPhysicalDeviceProperties(pd, props);
                    names.add(props.deviceNameString());
                }
                return names;
            } finally {
                vkDestroyInstance(inst, null);
            }
        }
    }

    // ---------------------------------------------------------------- instance / device selection

    private static VkInstance createInstance(MemoryStack stack) {
        VkApplicationInfo app = VkApplicationInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_APPLICATION_INFO)
                .pApplicationName(stack.UTF8("formicapi"))
                .applicationVersion(1)
                .pEngineName(stack.UTF8("formicapi"))
                .engineVersion(1)
                .apiVersion(VK_API_VERSION_1_2);

        VkInstanceCreateInfo info = VkInstanceCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO)
                .pApplicationInfo(app);

        if (Boolean.getBoolean("formicapi.vulkan.validation"))
            info.ppEnabledLayerNames(stack.pointers(stack.UTF8("VK_LAYER_KHRONOS_validation")));

        PointerBuffer pInstance = stack.mallocPointer(1);
        check(vkCreateInstance(info, null, pInstance), "vkCreateInstance (is a Vulkan 1.2 loader/driver installed?)");
        return new VkInstance(pInstance.get(0), info);
    }

    private static VkPhysicalDevice[] enumerate(MemoryStack stack, VkInstance instance) {
        IntBuffer count = stack.mallocInt(1);
        check(vkEnumeratePhysicalDevices(instance, count, null), "vkEnumeratePhysicalDevices");
        if (count.get(0) == 0)
            return new VkPhysicalDevice[0];

        PointerBuffer handles = stack.mallocPointer(count.get(0));
        check(vkEnumeratePhysicalDevices(instance, count, handles), "vkEnumeratePhysicalDevices");

        VkPhysicalDevice[] out = new VkPhysicalDevice[count.get(0)];
        for (int i = 0; i < out.length; i++)
            out[i] = new VkPhysicalDevice(handles.get(i), instance);
        return out;
    }

    private static int typeRank(int deviceType) {
        return switch (deviceType) {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU -> 4;
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU -> 3;
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU -> 2;
            case VK_PHYSICAL_DEVICE_TYPE_CPU -> 1;
            default -> 0;
        };
    }

    /** Null if usable, otherwise a short reason. */
    private static @Nullable String unusableReason(MemoryStack stack, VkPhysicalDevice pd) {
        try (MemoryStack inner = stack.push()) {
            VkPhysicalDeviceProperties props = VkPhysicalDeviceProperties.calloc(inner);
            vkGetPhysicalDeviceProperties(pd, props);
            if (props.apiVersion() < VK_API_VERSION_1_2)
                return "only supports Vulkan " + VK_VERSION_MAJOR(props.apiVersion()) + "." + VK_VERSION_MINOR(props.apiVersion()) + " (1.2 required)";
            if (findComputeQueueFamilyOrNeg(inner, pd) < 0)
                return "has no compute queue";
            VkPhysicalDeviceFeatures2 f = VkPhysicalDeviceFeatures2.calloc(inner).sType(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2);
            vkGetPhysicalDeviceFeatures2(pd, f);
            if (!f.features().shaderFloat64())
                return "doesn't support shaderFloat64 (double precision)";
            return null;
        }
    }

    private static VkPhysicalDevice selectDefault(MemoryStack stack, VkPhysicalDevice[] all) {
        VkPhysicalDevice best = null;
        int bestRank = -1;
        for (VkPhysicalDevice pd : all) {
            if (unusableReason(stack, pd) != null)
                continue;
            try (MemoryStack inner = stack.push()) {
                VkPhysicalDeviceProperties props = VkPhysicalDeviceProperties.calloc(inner);
                vkGetPhysicalDeviceProperties(pd, props);
                int rank = typeRank(props.deviceType());
                if (rank > bestRank) {
                    best = pd;
                    bestRank = rank;
                }
            }
        }
        if (best == null)
            throw new RuntimeException("No Vulkan 1.2 device with compute and shaderFloat64 (double precision) support found");
        return best;
    }

    private static VkPhysicalDevice requireUsable(MemoryStack stack, VkPhysicalDevice[] all, int index) {
        if (index >= all.length)
            throw new IllegalArgumentException("Vulkan device index " + index + " out of range (0.." + (all.length - 1) + ")");
        String reason = unusableReason(stack, all[index]);
        if (reason != null)
            throw new RuntimeException("Vulkan device #" + index + " " + reason);
        return all[index];
    }

    private static int findComputeQueueFamilyOrNeg(MemoryStack stack, VkPhysicalDevice pd) {
        IntBuffer count = stack.mallocInt(1);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, count, null);
        VkQueueFamilyProperties.Buffer families = VkQueueFamilyProperties.calloc(count.get(0), stack);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, count, families);

        // prefer a compute-only family (async compute queue) when it exists, like most compute frameworks do
        int fallback = -1;
        for (int i = 0; i < families.capacity(); i++) {
            int flags = families.get(i).queueFlags();
            if ((flags & VK_QUEUE_COMPUTE_BIT) == 0)
                continue;
            if ((flags & VK_QUEUE_GRAPHICS_BIT) == 0)
                return i;
            if (fallback < 0)
                fallback = i;
        }
        return fallback;
    }

    private static int findComputeQueueFamily(MemoryStack stack, VkPhysicalDevice pd) {
        int family = findComputeQueueFamilyOrNeg(stack, pd);
        if (family < 0)
            throw new RuntimeException("Vulkan device has no compute queue family");
        return family;
    }

    // ---------------------------------------------------------------- package-private plumbing used by VulkanKernel / VulkanResource

    VkDevice device() {
        return device;
    }

    static void check(int result, String what) {
        if (result != VK_SUCCESS)
            throw new RuntimeException(what + " failed with VkResult " + result);
    }

    private VulkanKernel registerKernel(String name, String glsl, String... features) {
        VulkanKernel k = new VulkanKernel(this, name, glsl, features);
        synchronized (lock) {
            kernels.add(k);
        }
        return k;
    }

    // ---------------------------------------------------------------- GpuExecutor: capabilities / kernels

    private boolean supportsFeature(String feature) {
        return switch (feature) {
            case "shaderFloat64" -> true; // required at selection
            case "shaderInt64" -> hasInt64;
            case "shaderBufferInt64Atomics" -> hasInt64Atomics;
            case "storageBuffer8BitAccess" -> has8BitStorage;
            case "shaderInt8" -> hasInt8;
            default -> false;
        };
    }

    @Override
    public boolean supports(Kernel kernel) {
        if (kernel instanceof PortableKernel portable) {
            if (portable.glslSource() == null)
                return false;
            for (String feature : portable.glslRequirements())
                if (!supportsFeature(feature))
                    return false;
            return true;
        }
        return kernel instanceof VulkanKernel;
    }

    @Override
    public Kernel createKernel(PortableKernel definition) {
        String source = definition.glslSource();
        if (source == null)
            throw new UnsupportedOperationException("Kernel '" + definition.name() + "' has no GLSL source, so it can't run on Vulkan");
        for (String feature : definition.glslRequirements())
            if (!supportsFeature(feature))
                throw new UnsupportedOperationException("Kernel '" + definition.name() + "' requires Vulkan feature '" + feature
                        + "', which '" + deviceName + "' doesn't provide");
        return registerKernel(definition.name(), source, definition.glslRequirements());
    }

    @Override
    public long maxWorkGroupSize() {
        return Math.min(maxInvocations, maxGroupSizeX);
    }

    @Override
    public GpuResource scalar(int value) {
        return VulkanResource.of(value);
    }

    @Override
    public GpuResource scalar(double value) {
        return VulkanResource.of(value);
    }

    @Override
    public GpuResource localMemory(long bytes) {
        return VulkanResource.local(bytes);
    }

    // ---------------------------------------------------------------- memory

    private int findMemoryType(int typeBits, int required, int preferred) {
        for (int pass = 0; pass < 2; pass++) {
            int want = pass == 0 ? (required | preferred) : required;
            for (int i = 0; i < memoryProperties.memoryTypeCount(); i++) {
                if ((typeBits & (1 << i)) != 0 && (memoryProperties.memoryTypes(i).propertyFlags() & want) == want)
                    return i;
            }
        }
        throw new RuntimeException("No suitable Vulkan memory type (required flags 0x" + Integer.toHexString(required) + ")");
    }

    /** @return {buffer, memory} */
    private long[] createBuffer(long sizeBytes, int usage, int requiredProps, int preferredProps) {
        try (MemoryStack stack = stackPush()) {
            VkBufferCreateInfo info = VkBufferCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO)
                    .size(sizeBytes)
                    .usage(usage)
                    .sharingMode(VK_SHARING_MODE_EXCLUSIVE);
            LongBuffer pBuffer = stack.mallocLong(1);
            check(vkCreateBuffer(device, info, null, pBuffer), "vkCreateBuffer(" + sizeBytes + " bytes)");
            long buffer = pBuffer.get(0);

            VkMemoryRequirements req = VkMemoryRequirements.calloc(stack);
            vkGetBufferMemoryRequirements(device, buffer, req);

            VkMemoryAllocateInfo alloc = VkMemoryAllocateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO)
                    .allocationSize(req.size())
                    .memoryTypeIndex(findMemoryType(req.memoryTypeBits(), requiredProps, preferredProps));
            LongBuffer pMemory = stack.mallocLong(1);
            int r = vkAllocateMemory(device, alloc, null, pMemory);
            if (r != VK_SUCCESS) {
                vkDestroyBuffer(device, buffer, null);
                check(r, "vkAllocateMemory(" + req.size() + " bytes)");
            }
            long memory = pMemory.get(0);
            check(vkBindBufferMemory(device, buffer, memory, 0), "vkBindBufferMemory");
            return new long[]{buffer, memory};
        }
    }

    private GpuResource allocate(long bytes) {
        // whole-word sizes: vkCmdFillBuffer / vkCmdUpdateBuffer work in 4-byte units
        long size = Math.max(4L, (bytes + 3L) & ~3L);
        synchronized (lock) {
            long[] bm = createBuffer(size, BUFFER_USAGE, 0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            return VulkanResource.buffer(this, bm[0], bm[1], size);
        }
    }

    @Override
    public GpuResource allocateDoubleBuffer(int length) {
        return allocate((long) length * Double.BYTES);
    }

    @Override
    public GpuResource allocateIntBuffer(int length) {
        return allocate((long) length * Integer.BYTES);
    }

    @Override
    public GpuResource allocateByteBuffer(int length) {
        return allocate(length);
    }

    /** Called by {@link VulkanResource#release()}: destroys now if idle, otherwise once every already-submitted command has finished. */
    void retire(long buffer, long memory) {
        synchronized (lock) {
            if (closed) return;
            if (inFlight.isEmpty()) {
                destroyBuffer(buffer, memory);
            } else {
                graveyard.add(new Dead(buffer, memory, submitSerial));
            }
        }
    }

    private void destroyBuffer(long buffer, long memory) {
        vkDestroyBuffer(device, buffer, null);
        vkFreeMemory(device, memory, null);
    }

    private void ensureStaging(long needed) {
        long want = Math.min(Math.max(needed, INITIAL_STAGING_BYTES), MAX_STAGING_BYTES);
        if (stagingCapacity >= want)
            return;

        long cap = Math.max(INITIAL_STAGING_BYTES, Long.highestOneBit(want - 1) << 1); // next power of two >= want
        cap = Math.min(cap, MAX_STAGING_BYTES);

        // every transfer that used the staging buffer was blocking, and nothing else touches it, so it's idle here
        releaseStaging();

        long[] bm = createBuffer(cap, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        stagingBuffer = bm[0];
        stagingMemory = bm[1];
        stagingCapacity = cap;
        try (MemoryStack stack = stackPush()) {
            PointerBuffer pData = stack.mallocPointer(1);
            check(vkMapMemory(device, stagingMemory, 0, VK_WHOLE_SIZE, 0, pData), "vkMapMemory(staging)");
            stagingPtr = pData.get(0);
        }
    }

    private void releaseStaging() {
        if (stagingBuffer == VK_NULL_HANDLE)
            return;
        vkUnmapMemory(device, stagingMemory);
        destroyBuffer(stagingBuffer, stagingMemory);
        stagingBuffer = VK_NULL_HANDLE;
        stagingMemory = VK_NULL_HANDLE;
        stagingPtr = 0;
        stagingCapacity = 0;
    }

    private ByteBuffer stagingView(int bytes) {
        return MemoryUtil.memByteBuffer(stagingPtr, bytes).order(ByteOrder.nativeOrder());
    }

    // ---------------------------------------------------------------- submission machinery

    @FunctionalInterface
    private interface Recorder {
        void record(VkCommandBuffer cb, MemoryStack stack);
    }

    private Submission acquireSubmission(MemoryStack stack) {
        reap();
        Submission s = idle.pollFirst();
        if (s != null) {
            check(vkResetFences(device, s.fence), "vkResetFences");
            check(vkResetCommandBuffer(s.cb, 0), "vkResetCommandBuffer");
            return s;
        }

        VkCommandBufferAllocateInfo alloc = VkCommandBufferAllocateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO)
                .commandPool(commandPool)
                .level(VK_COMMAND_BUFFER_LEVEL_PRIMARY)
                .commandBufferCount(1);
        PointerBuffer pCb = stack.mallocPointer(1);
        check(vkAllocateCommandBuffers(device, alloc, pCb), "vkAllocateCommandBuffers");

        VkFenceCreateInfo fenceInfo = VkFenceCreateInfo.calloc(stack).sType(VK_STRUCTURE_TYPE_FENCE_CREATE_INFO);
        LongBuffer pFence = stack.mallocLong(1);
        check(vkCreateFence(device, fenceInfo, null, pFence), "vkCreateFence");

        return new Submission(new VkCommandBuffer(pCb.get(0), device), pFence.get(0));
    }

    /**
     * Records {@code recorder}'s commands behind a global barrier, submits them, and optionally waits.
     * {@code descriptorSet} (0 if none) is freed once this submission completes.
     */
    private void submit(Recorder recorder, long descriptorSet, boolean wait) {
        try (MemoryStack stack = stackPush()) {
            Submission s = acquireSubmission(stack);

            VkCommandBufferBeginInfo begin = VkCommandBufferBeginInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
                    .flags(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
            check(vkBeginCommandBuffer(s.cb, begin), "vkBeginCommandBuffer");

            // Execution + memory dependency on everything submitted before (same queue, submission order):
            // this is what makes the queue behave like an in-order one. HOST is in the source scope so
            // staging writes made on the host before submit are visible to the transfer that reads them.
            VkMemoryBarrier.Buffer barrier = VkMemoryBarrier.calloc(1, stack);
            barrier.get(0)
                    .sType(VK_STRUCTURE_TYPE_MEMORY_BARRIER)
                    .srcAccessMask(VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT)
                    .dstAccessMask(VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
                            | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
            vkCmdPipelineBarrier(s.cb,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                    0, barrier, null, null);

            recorder.record(s.cb, stack);
            check(vkEndCommandBuffer(s.cb), "vkEndCommandBuffer");

            VkSubmitInfo submit = VkSubmitInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_SUBMIT_INFO)
                    .pCommandBuffers(stack.pointers(s.cb));
            check(vkQueueSubmit(queue, submit, s.fence), "vkQueueSubmit");

            s.serial = ++submitSerial;
            if (descriptorSet != VK_NULL_HANDLE)
                s.descriptorSets.add(descriptorSet);
            inFlight.addLast(s);

            if (wait)
                waitFor(s);
        }
    }

    /** Waits for {@code target} and everything submitted before it, then retires them. */
    private void waitFor(Submission target) {
        try (MemoryStack stack = stackPush()) {
            for (Submission s : inFlight) {
                check(vkWaitForFences(device, stack.longs(s.fence), true, Long.MAX_VALUE), "vkWaitForFences");
                if (s == target)
                    break;
            }
        }
        reap();
    }

    private void waitAll() {
        Submission last = inFlight.peekLast();
        if (last != null)
            waitFor(last);
    }

    /** Retires every finished submission (oldest first), freeing their descriptor sets and any buffers waiting on them. */
    private void reap() {
        while (!inFlight.isEmpty()) {
            Submission s = inFlight.peekFirst();
            int status = vkGetFenceStatus(device, s.fence);
            if (status == VK_NOT_READY)
                break;
            check(status, "vkGetFenceStatus");

            inFlight.pollFirst();
            for (long set : s.descriptorSets)
                vkFreeDescriptorSets(device, descriptorPool, set);
            s.descriptorSets.clear();
            completedSerial = Math.max(completedSerial, s.serial);
            idle.addLast(s);
        }

        if (!graveyard.isEmpty()) {
            graveyard.removeIf(dead -> {
                if (inFlight.isEmpty() || dead.serial() <= completedSerial) {
                    destroyBuffer(dead.buffer(), dead.memory());
                    return true;
                }
                return false;
            });
        }
    }

    // ---------------------------------------------------------------- dispatch

    void dispatch(VulkanKernel kernel, int globalSize, int localSize, GpuResource[] args, boolean blocking) {
        if (globalSize <= 0) return;
        if (localSize < 0 || (localSize > 0 && globalSize % localSize != 0))
            throw new IllegalArgumentException("globalSize (" + globalSize + ") must be a multiple of localSize (" + localSize + ")");

        final int local = localSize > 0 ? localSize : defaultLocalSize;
        if (local > maxWorkGroupSize())
            throw new IllegalArgumentException("localSize " + local + " exceeds this device's maximum work-group size " + maxWorkGroupSize());

        // gather buffer bindings + pack scalars into the push-constant block
        long[] buffers = new long[kernel.bindingCount];
        int bufferCount = 0;
        ByteBuffer push = MemoryUtil.memCalloc(VulkanKernel.PUSH_BYTES).order(ByteOrder.nativeOrder());
        try {
            push.putInt(0, globalSize); // implicit `int gsz`
            int offset = Integer.BYTES;

            for (int i = 0; i < args.length; i++) {
                if (args[i] == null)
                    throw new NullPointerException("Kernel arg was null at index: " + i);
                if (!(args[i] instanceof VulkanResource arg))
                    throw new RuntimeException("Invalid GpuResource implementation for Vulkan: " + args[i].getClass());

                switch (arg.kind) {
                    case BUFFER -> {
                        if (bufferCount >= buffers.length)
                            throw new IllegalArgumentException("Kernel '" + kernel.name() + "' declares " + buffers.length
                                    + " buffer bindings but was given more buffer arguments");
                        buffers[bufferCount++] = arg.buffer;
                    }
                    case SCALAR -> {
                        offset = (offset + arg.width - 1) & -arg.width; // std430 scalar alignment == its size
                        if (offset + arg.width > VulkanKernel.PUSH_BYTES)
                            throw new IllegalArgumentException("Kernel '" + kernel.name() + "': scalar arguments exceed the " + VulkanKernel.PUSH_BYTES + "-byte push-constant limit");
                        if (arg.width == Integer.BYTES) push.putInt(offset, (int) arg.bits);
                        else push.putLong(offset, arg.bits);
                        offset += arg.width;
                    }
                    case LOCAL -> { /* shared memory is declared in the shader */ }
                }
            }
            if (bufferCount != buffers.length)
                throw new IllegalArgumentException("Kernel '" + kernel.name() + "' declares " + buffers.length
                        + " buffer bindings but was given " + bufferCount + " buffer arguments");

            final int pushBytes = (offset + 3) & ~3;
            push.limit(pushBytes);

            long groups = (globalSize + (long) local - 1) / local;
            final int gx = (int) Math.min(groups, maxGroupCountX);
            long gyLong = (groups + gx - 1) / gx;
            if (gyLong > maxGroupCountY)
                throw new IllegalArgumentException("globalSize " + globalSize + " needs more work-groups than this device can dispatch");
            final int gy = (int) gyLong;

            synchronized (lock) {
                if (closed) throw new IllegalStateException("VulkanGpuExecutor is closed");
                long pipeline = kernel.pipeline(local);
                long set = allocateDescriptorSet(kernel.descriptorSetLayout);

                try (MemoryStack stack = stackPush()) {
                    VkDescriptorBufferInfo.Buffer infos = VkDescriptorBufferInfo.calloc(buffers.length, stack);
                    VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(buffers.length, stack);
                    for (int i = 0; i < buffers.length; i++) {
                        infos.get(i).buffer(buffers[i]).offset(0).range(VK_WHOLE_SIZE);
                        writes.get(i)
                                .sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET)
                                .dstSet(set)
                                .dstBinding(i)
                                .dstArrayElement(0)
                                .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                                .pBufferInfo(infos.slice(i, 1));
                    }
                    vkUpdateDescriptorSets(device, writes, null);
                }

                submit((cb, stack) -> {
                    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
                    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.pipelineLayout, 0, stack.longs(set), null);
                    vkCmdPushConstants(cb, kernel.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push);
                    vkCmdDispatch(cb, gx, gy, 1);
                }, set, blocking);
            }
        } finally {
            MemoryUtil.memFree(push);
        }
    }

    private long allocateDescriptorSet(long layout) {
        try (MemoryStack stack = stackPush()) {
            VkDescriptorSetAllocateInfo info = VkDescriptorSetAllocateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO)
                    .descriptorPool(descriptorPool)
                    .pSetLayouts(stack.longs(layout));
            LongBuffer pSet = stack.mallocLong(1);

            int r = vkAllocateDescriptorSets(device, info, pSet);
            if (r != VK_SUCCESS) {
                // pool exhausted by in-flight work: let it drain (which frees their sets) and try once more
                waitAll();
                r = vkAllocateDescriptorSets(device, info, pSet);
            }
            check(r, "vkAllocateDescriptorSets");
            return pSet.get(0);
        }
    }

    // ---------------------------------------------------------------- host transfers (blocking)

    @FunctionalInterface
    private interface StageWriter {
        /** Fill {@code staging[0, byteCount)} with host bytes {@code [hostBytePos, hostBytePos + byteCount)} of the transfer. */
        void write(ByteBuffer staging, long hostBytePos, int byteCount);
    }

    @FunctionalInterface
    private interface StageReader {
        /** Consume {@code staging[0, byteCount)} as host bytes {@code [hostBytePos, hostBytePos + byteCount)} of the transfer. */
        void read(ByteBuffer staging, long hostBytePos, int byteCount);
    }

    private static VulkanResource res(GpuResource r) {
        if (!(r instanceof VulkanResource v) || !v.isBuffer())
            throw new RuntimeException("Invalid GpuResource implementation for Vulkan: " + r.getClass());
        return v;
    }

    private void write(GpuResource dst, long dstByteOffset, long totalBytes, StageWriter writer) {
        if (totalBytes <= 0) return;
        VulkanResource target = res(dst);
        synchronized (lock) {
            long pos = 0;
            while (pos < totalBytes) {
                int chunk = (int) Math.min(totalBytes - pos, MAX_STAGING_BYTES);
                ensureStaging(chunk);
                writer.write(stagingView(chunk), pos, chunk);

                final long dstOffset = dstByteOffset + pos;
                final int bytes = chunk;
                submit((cb, stack) -> {
                    VkBufferCopy.Buffer region = VkBufferCopy.calloc(1, stack).srcOffset(0).dstOffset(dstOffset).size(bytes);
                    vkCmdCopyBuffer(cb, stagingBuffer, target.buffer, region);
                }, VK_NULL_HANDLE, true);
                pos += chunk;
            }
        }
    }

    private void read(GpuResource src, long srcByteOffset, long totalBytes, StageReader reader) {
        if (totalBytes <= 0) return;
        VulkanResource source = res(src);
        synchronized (lock) {
            long pos = 0;
            while (pos < totalBytes) {
                int chunk = (int) Math.min(totalBytes - pos, MAX_STAGING_BYTES);
                ensureStaging(chunk);

                final long srcOffset = srcByteOffset + pos;
                final int bytes = chunk;
                submit((cb, stack) -> {
                    VkBufferCopy.Buffer region = VkBufferCopy.calloc(1, stack).srcOffset(srcOffset).dstOffset(0).size(bytes);
                    vkCmdCopyBuffer(cb, source.buffer, stagingBuffer, region);

                    // make the transfer's writes available to the host
                    VkMemoryBarrier.Buffer toHost = VkMemoryBarrier.calloc(1, stack);
                    toHost.get(0)
                            .sType(VK_STRUCTURE_TYPE_MEMORY_BARRIER)
                            .srcAccessMask(VK_ACCESS_TRANSFER_WRITE_BIT)
                            .dstAccessMask(VK_ACCESS_HOST_READ_BIT);
                    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, toHost, null, null);
                }, VK_NULL_HANDLE, true);

                reader.read(stagingView(chunk), pos, chunk);
                pos += chunk;
            }
        }
    }

    // ---- doubles

    @Override
    public void uploadDoubles(GpuResource buffer, double[] host, int length) {
        uploadDoubles(buffer, 0, host, 0, length);
    }

    @Override
    public void uploadDoubles(GpuResource buffer, int elementOffset, double[] host, int hostOffset, int length) {
        write(buffer, (long) elementOffset * Double.BYTES, (long) length * Double.BYTES,
                (staging, pos, n) -> staging.asDoubleBuffer().put(host, hostOffset + (int) (pos / Double.BYTES), n / Double.BYTES));
    }

    @Override
    public void downloadDoubles(GpuResource buffer, double[] host, int length) {
        downloadDoubles(buffer, 0, host, length);
    }

    @Override
    public void downloadDoubles(GpuResource buffer, int elementOffset, double[] host, int length) {
        read(buffer, (long) elementOffset * Double.BYTES, (long) length * Double.BYTES,
                (staging, pos, n) -> staging.asDoubleBuffer().get(host, (int) (pos / Double.BYTES), n / Double.BYTES));
    }

    @Override
    public void uploadDoubleAt(GpuResource buffer, int elementOffset, double value) {
        write(buffer, (long) elementOffset * Double.BYTES, Double.BYTES, (staging, pos, n) -> staging.putDouble(0, value));
    }

    @Override
    public double downloadDoubleAt(GpuResource buffer, int elementOffset) {
        double[] out = new double[1];
        read(buffer, (long) elementOffset * Double.BYTES, Double.BYTES, (staging, pos, n) -> out[0] = staging.getDouble(0));
        return out[0];
    }

    // ---- ints

    @Override
    public void uploadInts(GpuResource buffer, int[] host, int length) {
        uploadInts(buffer, 0, host, 0, length);
    }

    @Override
    public void uploadInts(GpuResource buffer, int elementOffset, int[] host, int hostOffset, int length) {
        write(buffer, (long) elementOffset * Integer.BYTES, (long) length * Integer.BYTES,
                (staging, pos, n) -> staging.asIntBuffer().put(host, hostOffset + (int) (pos / Integer.BYTES), n / Integer.BYTES));
    }

    @Override
    public void downloadInts(GpuResource buffer, int[] host, int length) {
        downloadInts(buffer, 0, host, length);
    }

    @Override
    public void downloadInts(GpuResource buffer, int elementOffset, int[] host, int length) {
        read(buffer, (long) elementOffset * Integer.BYTES, (long) length * Integer.BYTES,
                (staging, pos, n) -> staging.asIntBuffer().get(host, (int) (pos / Integer.BYTES), n / Integer.BYTES));
    }

    @Override
    public void uploadIntAt(GpuResource buffer, int elementOffset, int value) {
        write(buffer, (long) elementOffset * Integer.BYTES, Integer.BYTES, (staging, pos, n) -> staging.putInt(0, value));
    }

    @Override
    public int downloadIntAt(GpuResource buffer, int elementOffset) {
        int[] out = new int[1];
        read(buffer, (long) elementOffset * Integer.BYTES, Integer.BYTES, (staging, pos, n) -> out[0] = staging.getInt(0));
        return out[0];
    }

    // ---- bytes

    @Override
    public void uploadBytes(GpuResource buffer, byte[] host, int length) {
        write(buffer, 0, length, (staging, pos, n) -> staging.put(host, (int) pos, n));
    }

    @Override
    public void downloadBytes(GpuResource buffer, byte[] host, int length) {
        read(buffer, 0, length, (staging, pos, n) -> staging.get(host, (int) pos, n));
    }

    @Override
    public void uploadByteAt(GpuResource buffer, int elementOffset, byte value) {
        write(buffer, elementOffset, 1, (staging, pos, n) -> staging.put(0, value));
    }

    @Override
    public byte downloadByteAt(GpuResource buffer, int elementOffset) {
        byte[] out = new byte[1];
        read(buffer, elementOffset, 1, (staging, pos, n) -> out[0] = staging.get(0));
        return out[0];
    }

    // ---------------------------------------------------------------- copy / fill (non-blocking)

    @Override
    public void copyDoubleBuffer(GpuResource src, int srcOffset, GpuResource dst, int dstOffset, int length) {
        copy(src, (long) srcOffset * Double.BYTES, dst, (long) dstOffset * Double.BYTES, (long) length * Double.BYTES);
    }

    @Override
    public void copyIntBuffer(GpuResource src, int srcOffset, GpuResource dst, int dstOffset, int length) {
        copy(src, (long) srcOffset * Integer.BYTES, dst, (long) dstOffset * Integer.BYTES, (long) length * Integer.BYTES);
    }

    @Override
    public void copyByteBuffer(GpuResource src, int srcOffset, GpuResource dst, int dstOffset, int length) {
        copy(src, srcOffset, dst, dstOffset, length);
    }

    private void copy(GpuResource src, long srcByteOffset, GpuResource dst, long dstByteOffset, long bytes) {
        if (bytes <= 0) return;
        VulkanResource s = res(src);
        VulkanResource d = res(dst);
        synchronized (lock) {
            submit((cb, stack) -> {
                VkBufferCopy.Buffer region = VkBufferCopy.calloc(1, stack).srcOffset(srcByteOffset).dstOffset(dstByteOffset).size(bytes);
                vkCmdCopyBuffer(cb, s.buffer, d.buffer, region);
            }, VK_NULL_HANDLE, false);
        }
    }

    private void fill32(GpuResource buffer, long byteOffset, long bytes, int pattern) {
        VulkanResource b = res(buffer);
        synchronized (lock) {
            submit((cb, stack) -> vkCmdFillBuffer(cb, b.buffer, byteOffset, bytes, pattern), VK_NULL_HANDLE, false);
        }
    }

    @Override
    public void fillDoubleBuffer(GpuResource buffer, int offset, double value, int length) {
        if (length <= 0) return;
        long bits = Double.doubleToRawLongBits(value);
        int lo = (int) bits;
        int hi = (int) (bits >>> 32);
        if (lo == hi) { // includes 0.0: expressible as a repeated 32-bit pattern
            fill32(buffer, (long) offset * Double.BYTES, (long) length * Double.BYTES, lo);
            return;
        }
        dispatch(fillDoubleKernel, length, 0, new GpuResource[]{buffer, scalar(offset), scalar(value)}, false);
    }

    @Override
    public void fillIntBuffer(GpuResource buffer, int offset, int value, int length) {
        if (length <= 0) return;
        fill32(buffer, (long) offset * Integer.BYTES, (long) length * Integer.BYTES, value);
    }

    /**
     * Byte fill. {@code vkCmdFillBuffer} only works on whole, aligned 32-bit
     * words, so the aligned middle goes through it and the (at most 3-byte)
     * unaligned head and tail are written through the staging path.
     */
    @Override
    public void fillByteBuffer(GpuResource buffer, int offset, byte value, int length) {
        if (length <= 0) return;
        long start = offset;
        long end = start + length;
        long alignedStart = (start + 3) & ~3L;
        long alignedEnd = end & ~3L;

        if (alignedStart >= alignedEnd) {
            writeConstantBytes(buffer, start, (int) (end - start), value);
            return;
        }
        if (alignedStart > start)
            writeConstantBytes(buffer, start, (int) (alignedStart - start), value);

        int b = value & 0xFF;
        fill32(buffer, alignedStart, alignedEnd - alignedStart, b * 0x01010101);

        if (end > alignedEnd)
            writeConstantBytes(buffer, alignedEnd, (int) (end - alignedEnd), value);
    }

    private void writeConstantBytes(GpuResource buffer, long byteOffset, int count, byte value) {
        write(buffer, byteOffset, count, (staging, pos, n) -> {
            for (int i = 0; i < n; i++)
                staging.put(i, value);
        });
    }

    // ---------------------------------------------------------------- lifecycle

    /** Blocks until every previously submitted operation has completed. */
    @Override
    public void finish() {
        synchronized (lock) {
            check(vkQueueWaitIdle(queue), "vkQueueWaitIdle");
            reap();
        }
    }

    @Override
    public void close() {
        synchronized (lock) {
            if (closed) return;
            vkDeviceWaitIdle(device);
            reap();
            closed = true;
            PortableKernel.forgetExecutor(this);

            for (VulkanKernel k : kernels)
                k.destroy();
            kernels.clear();

            for (Dead dead : graveyard)
                destroyBuffer(dead.buffer(), dead.memory());
            graveyard.clear();
            releaseStaging();

            for (Submission s : idle)
                vkDestroyFence(device, s.fence, null);
            idle.clear();

            vkDestroyDescriptorPool(device, descriptorPool, null);
            vkDestroyCommandPool(device, commandPool, null); // frees its command buffers
            memoryProperties.free();
            vkDestroyDevice(device, null);
            vkDestroyInstance(instance, null);
        }
    }
}