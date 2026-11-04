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
import java.util.Arrays;
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
 * <h1>Execution model</h1>
 * <ul>
 *   <li><b>Serials and fences.</b> Every operation is its own one-time
 *       command buffer submitted to one queue with its own fence, and gets the
 *       next value of a monotonically increasing serial. A serial is this
 *       backend's {@link VulkanEvent}; finished submissions are retired oldest
 *       first, which yields a "completed up to serial N" watermark, and the
 *       host can wait on any single serial through that submission's fence.</li>
 *   <li><b>Per-resource dependencies, like OpenCL's event wait lists.</b> Each
 *       buffer remembers the serial of the last operation that touched it
 *       ({@link VulkanResource#pendingEvents()}). A submission opens with a
 *       memory barrier <em>only if</em> one of the buffers it touches still has
 *       unfinished work that no earlier barrier already covers. Operations on
 *       independent buffers get no barrier at all and may overlap on the
 *       device; operations sharing a buffer stay ordered. Waiting is always
 *       device-side -- the host never blocks for a dependency.</li>
 *   <li><b>Non-blocking transfers in.</b> Uploads copy the host data into a
 *       pooled, persistently mapped staging block (so the host array is free
 *       to reuse on return), submit the copy and return. The block is
 *       recycled when that submission completes. A later dispatch on the
 *       same buffer is ordered after the upload by the dependency above; a
 *       dispatch on any other buffer is not delayed by it.</li>
 *   <li><b>Blocking only where it must.</b> Downloads and {@code useBlocking}
 *       wait for their own serial (outside the executor lock), nothing more.
 *       Dispatches, copies, fills and uploads return immediately.</li>
 *   <li><b>Deferred destruction.</b> {@link GpuResource#release()} on a buffer
 *       can't free it while queued work still uses it, so destruction is
 *       queued behind that buffer's own last serial and carried out once the
 *       completion watermark reaches it.</li>
 *   <li><b>Locking.</b> One queue means external synchronization: submission
 *       and bookkeeping happen under one monitor; host waits happen outside it.</li>
 * </ul>
 *
 * <h3>Requirements and limits</h3>
 * Vulkan 1.2. {@code shaderFloat64} is <em>optional</em>: devices without
 * it work for the float/int API, and only double-precision kernels are
 * refused (see {@link #supportsFeature}); the double helpers throw
 * {@link UnsupportedOperationException} there. 64-bit buffer
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
    /** Transfers larger than this are split into several chunks. Multiple of 8 so a chunk never splits a double. */
    static final  long MAX_STAGING_BYTES           = 64L << 20;
    /** Cap on total staging memory (free + in flight). Past it, an upload waits for earlier ones to drain instead of allocating more. */
    private static final long MAX_STAGING_TOTAL    = 256L << 20;
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
    private final boolean hasInt32;
    private final boolean hasInt16;
    private final boolean hasInt8;
    private final boolean hasFloat64;
    private final boolean hasFloat32;
    private final boolean hasFloat16;
    private final boolean hasFloat8;

    private final boolean hasInt64Atomics;
    private final boolean has8BitStorage;

    private final int  maxInvocations;
    private final int  maxGroupSizeX;
    private final int  maxGroupCountX;
    private final int  maxGroupCountY;
    private final int  defaultLocalSize;

    private final long commandPool;
    private final long descriptorPool;

    // submission tracking: one serial per queue submission, each with its own fence
    private static final class Submission {
        final VkCommandBuffer cb;
        final long fence;
        final List<Long> descriptorSets = new ArrayList<>(2);
        @Nullable Staging staging; // upload staging block, handed back to the pool once this submission completes
        long serial;
        int waiters; // host threads currently blocked on `fence`: it must not be reset/recycled until they leave (guarded by lock)

        Submission(VkCommandBuffer cb, long fence) {
            this.cb = cb;
            this.fence = fence;
        }
    }

    /** One persistently mapped host-visible block used to move bytes to/from the device. */
    private record Staging(long buffer, long memory, long ptr, long capacity) {
    }

    private record Dead(long buffer, long memory, long serial) {
    }

    private final ArrayDeque<Submission> inFlight = new ArrayDeque<>();
    private final ArrayDeque<Submission> idle = new ArrayDeque<>();
    private final List<Dead> graveyard = new ArrayList<>();
    private long submitSerial = 0;              // guarded by lock
    private volatile long completedSerial = 0;  // written under lock, readable anywhere
    /** Serial of the latest submission that opened with a barrier: it covers every submission before it. Guarded by lock. */
    private long barrierSerial = 0;

    // staging pool (guarded by lock)
    private final List<Staging> freeStaging = new ArrayList<>();
    private long stagingLive = 0;

    private final List<VulkanKernel> kernels = new ArrayList<>();
    private final @Nullable VulkanKernel fillDoubleKernel; // null without shaderFloat64
    private final VulkanKernel selfTestKernel;

    private volatile boolean closed = false;

    // optional validation-layer output (-Dformicapi.vulkan.validation=true)
    private @Nullable VkDebugUtilsMessengerCallbackEXT debugCallback;
    private long debugMessenger = VK_NULL_HANDLE;

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
            final boolean validation = Boolean.getBoolean("formicapi.vulkan.validation");
            this.instance = createInstance(stack, validation);
            if (validation)
                installDebugMessenger(stack);

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


            //TODO add more, in case we need features in the future (yes I know that this implementation is against SOLID principles)
            this.hasInt64 = supported.features().shaderInt64();//That's longs -> we use 4 * 8 = 32 bits integers
            this.hasInt32 = true;
            this.hasInt16 = supported.features().shaderInt16();
            this.hasInt8 = supported12.shaderInt8();// single byte ints ? who needs that ??
            this.hasFloat64 = supported.features().shaderFloat64();
            this.hasFloat32 = true;
            this.hasFloat16 = supported12.shaderFloat16();
            this.hasFloat8  = false;
            this.hasInt64Atomics = hasInt64 && supported12.shaderBufferInt64Atomics();
            this.has8BitStorage = supported12.storageBuffer8BitAccess();// that's for bools ? or directly bit's who needs that ??

            // enable exactly what we use and what's available
            VkPhysicalDeviceVulkan12Features enable12 = VkPhysicalDeviceVulkan12Features.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES)
                    .shaderBufferInt64Atomics(hasInt64Atomics)
                    .storageBuffer8BitAccess(has8BitStorage)
                    .shaderInt8(hasInt8);
            VkPhysicalDeviceFeatures2 enable = VkPhysicalDeviceFeatures2.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2)
                    .pNext(enable12.address());
            enable.features()
                    .shaderFloat64(hasFloat64)
                    .shaderInt64(hasInt64)
                    .shaderInt16(hasInt16);

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
        this.fillDoubleKernel = !hasFloat64 ? null : registerKernel("fill_double", """
                layout(std430, binding = 0) buffer D { double d[]; };
                layout(push_constant) uniform PC { int gsz; int offset; double value; } pc;
                void main() {
                    uint i = GID;
                    if (i >= uint(pc.gsz)) return;
                    d[uint(pc.offset) + i] = pc.value;
                }
                """, "shaderFloat64");

        // Prove compute actually works on this device before anyone relies on it: a wrong-but-silent
        // driver/setup problem otherwise surfaces much later as garbage results or VK_ERROR_DEVICE_LOST.
        this.selfTestKernel = registerKernel("selftest", """
                layout(std430, binding = 0) buffer O { float o[]; };
                layout(std430, binding = 1) buffer I { float src[]; };
                layout(std430, binding = 2) buffer X { int idx[]; };
                layout(push_constant) uniform PC { int gsz; int n; float a; } pc;
                void main() {
                    uint i = GID;
                    if (i >= uint(pc.gsz)) return;
                    o[i] = src[idx[i]] * pc.a + float(pc.n);
                }
                """);
        try {
            selfTest();
        } catch (RuntimeException e) {
            try {
                close();
            } catch (RuntimeException ignored) {
                // already failing; the original problem is the one worth reporting
            }
            throw e instanceof IllegalStateException ? e
                    : new IllegalStateException("Vulkan compute self-test failed on '" + deviceName + "': " + e.getMessage()
                    + " (re-run with -Dformicapi.vulkan.validation=true for validation-layer output)", e);
        }
    }

    private void selfTest() {
        GpuResource in = allocateFloatBuffer(4), idx = allocateIntBuffer(4), out = allocateFloatBuffer(4);
        try {
            uploadFloats(in, new float[]{10, 20, 30, 40}, 4);
            uploadInts(idx, new int[]{3, 2, 1, 0}, 4);
            fillFloatBuffer(out, 0, 0f, 4);
            // three buffers + an int and a float push constant + an indexed read: the shape of every real kernel
            dispatch(selfTestKernel, 4, 0, new GpuResource[]{out, in, idx, scalar(7), scalar(0.5f)}, true);

            float[] got = new float[4];
            downloadFloats(out, got, 4);
            float[] expected = {40 * 0.5f + 7, 30 * 0.5f + 7, 20 * 0.5f + 7, 10 * 0.5f + 7};
            if (!Arrays.equals(got, expected))
                throw new IllegalStateException("Vulkan compute self-test failed on '" + deviceName + "': expected "
                        + Arrays.toString(expected) + " but got " + Arrays.toString(got)
                        + " (re-run with -Dformicapi.vulkan.validation=true for validation-layer output)");
        } finally {
            in.release();
            idx.release();
            out.release();
        }
    }

    /** Human-readable name of the selected physical device. */
    public String deviceName() {
        return deviceName;
    }

    /** Names of every Vulkan physical device the driver exposes, in the order {@link #VulkanGpuExecutor(int)} indexes them. */
    public static List<String> deviceNames() {
        try (MemoryStack stack = stackPush()) {
            VkInstance inst = createInstance(stack, false);
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

    /** Routes validation-layer warnings/errors to {@code System.err}, so they show up in test and game logs. */
    private void installDebugMessenger(MemoryStack stack) {
        debugCallback = VkDebugUtilsMessengerCallbackEXT.create((severity, types, pCallbackData, pUserData) -> {
            VkDebugUtilsMessengerCallbackDataEXT data = VkDebugUtilsMessengerCallbackDataEXT.create(pCallbackData);
            System.err.println("[vulkan validation] " + data.pMessageString());
            return VK_FALSE;
        });
        VkDebugUtilsMessengerCreateInfoEXT info = VkDebugUtilsMessengerCreateInfoEXT.calloc(stack)
                .sType(EXTDebugUtils.VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT)
                .messageSeverity(EXTDebugUtils.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                        | EXTDebugUtils.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
                .messageType(EXTDebugUtils.VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                        | EXTDebugUtils.VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                        | EXTDebugUtils.VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT)
                .pfnUserCallback(debugCallback);
        LongBuffer pMessenger = stack.mallocLong(1);
        check(EXTDebugUtils.vkCreateDebugUtilsMessengerEXT(instance, info, null, pMessenger), "vkCreateDebugUtilsMessengerEXT");
        debugMessenger = pMessenger.get(0);
    }

    private static VkInstance createInstance(MemoryStack stack, boolean validation) {
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

        if (validation) {
            info.ppEnabledLayerNames(stack.pointers(stack.UTF8("VK_LAYER_KHRONOS_validation")));
            info.ppEnabledExtensionNames(stack.pointers(stack.UTF8(EXTDebugUtils.VK_EXT_DEBUG_UTILS_EXTENSION_NAME)));
        }

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
            throw new RuntimeException("No Vulkan 1.2 device with compute support found");
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
            case "shaderFloat64" -> hasFloat64;
            case "shaderFloat32" -> hasFloat32;//Always true
            case "shaderFloat16" -> hasFloat16;
            case "shaderFloat8" -> hasFloat8;
            case "shaderInt64" -> hasInt64;
            case "shaderInt32" -> hasInt32;//Always true
            case "shaderInt16" -> hasInt16;
            case "shaderInt8" -> hasInt8;
            case "shaderBufferInt64Atomics" -> hasInt64Atomics;
            case "storageBuffer8BitAccess" -> has8BitStorage;
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
    public GpuResource scalar(float value) {
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
    public GpuResource allocateFloatBuffer(int length) {
        return allocate((long) length * Float.BYTES);

    }

    @Override
    public GpuResource allocateIntBuffer(int length) {
        return allocate((long) length * Integer.BYTES);
    }

    @Override
    public GpuResource allocateByteBuffer(int length) {
        return allocate(length);
    }

    /**
     * Called by {@link VulkanResource#release()}: destroys now if the buffer's
     * last operation ({@code serial}, 0 if none) has already finished, otherwise
     * once the completion watermark reaches it.
     */
    void retire(long buffer, long memory, long serial) {
        synchronized (lock) {
            if (closed) return;
            reap();
            if (serial <= completedSerial)
                destroyBuffer(buffer, memory);
            else
                graveyard.add(new Dead(buffer, memory, serial));
        }
    }

    private void destroyBuffer(long buffer, long memory) {
        vkDestroyBuffer(device, buffer, null);
        vkFreeMemory(device, memory, null);
    }

    /**
     * True once the submission with this serial (and so everything before it) is known to have finished. Cheap and
     * safe from any thread; it reflects the last retirement pass, so it can lag the device slightly (never lead it).
     */
    boolean isComplete(long serial) {
        return serial <= completedSerial || closed;
    }

    /**
     * Blocks until {@code serial} has finished, then retires whatever that frees up. Needs no lock held: the wait
     * itself happens outside it, so other threads keep enqueuing. The target's fence is pinned (waiters) so a
     * concurrent retirement pass can't reset it while we are blocked on it.
     */
    private void waitAndReap(long serial) {
        if (serial <= 0 || serial <= completedSerial)
            return;
        Submission target = null;
        synchronized (lock) {
            if (closed) return;
            reap();
            if (serial <= completedSerial)
                return;
            for (Submission s : inFlight) {
                if (s.serial == serial) {
                    target = s;
                    break;
                }
            }
            if (target == null)
                throw new IllegalStateException("Submission " + serial + " is neither complete nor in flight");
            target.waiters++;
        }
        try {
            try (MemoryStack stack = stackPush()) {
                check(vkWaitForFences(device, stack.longs(target.fence), true, Long.MAX_VALUE), "vkWaitForFences");
            }
        } finally {
            synchronized (lock) {
                target.waiters--;
                if (!closed)
                    reap();
            }
        }
    }

    // ---- staging pool

    private @Nullable Staging takeFree(long needed) {
        Staging best = null;
        for (Staging s : freeStaging)
            if (s.capacity() >= needed && (best == null || s.capacity() < best.capacity()))
                best = s;
        if (best != null)
            freeStaging.remove(best);
        return best;
    }

    /** A staging block of at least {@code needed} bytes ({@code <= MAX_STAGING_BYTES}), reusing a free one when possible. Lock held. */
    private Staging acquireStaging(long needed) {
        Staging hit = takeFree(needed);
        if (hit != null)
            return hit;

        long cap = Math.max(INITIAL_STAGING_BYTES, Long.highestOneBit(needed - 1) << 1); // next power of two >= needed
        cap = Math.min(cap, MAX_STAGING_BYTES);

        if (stagingLive + cap > MAX_STAGING_TOTAL) {
            // backpressure: let in-flight uploads finish so their blocks come back, instead of growing without bound
            waitAndReap(submitSerial);
            hit = takeFree(needed);
            if (hit != null)
                return hit;
            for (Staging s : freeStaging) // idle blocks of the wrong size: free them to make room
                destroyStaging(s);
            freeStaging.clear();
        }
        return createStaging(cap);
    }

    private Staging createStaging(long cap) {
        long[] bm = createBuffer(cap, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        try (MemoryStack stack = stackPush()) {
            PointerBuffer pData = stack.mallocPointer(1);
            int r = vkMapMemory(device, bm[1], 0, VK_WHOLE_SIZE, 0, pData);
            if (r != VK_SUCCESS) {
                destroyBuffer(bm[0], bm[1]);
                check(r, "vkMapMemory(staging)");
            }
            stagingLive += cap;
            return new Staging(bm[0], bm[1], pData.get(0), cap);
        }
    }

    private void destroyStaging(Staging s) {
        vkUnmapMemory(device, s.memory());
        destroyBuffer(s.buffer(), s.memory());
        stagingLive -= s.capacity();
    }

    private static ByteBuffer view(Staging s, int bytes) {
        return MemoryUtil.memByteBuffer(s.ptr(), bytes).order(ByteOrder.nativeOrder());
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
     * Records {@code recorder}'s commands and submits them, returning the new submission's serial. Never blocks
     * (callers that need the result wait on the returned serial themselves). Lock held.
     *
     * <p>{@code touched} are the buffers this operation reads or writes. If any of them still has unfinished work
     * that no earlier barrier covers, the command buffer opens with a global memory barrier -- an execution +
     * memory dependency on everything submitted before it, the Vulkan equivalent of putting those buffers'
     * pending events in an OpenCL wait list. Otherwise no barrier is recorded and the work may overlap freely with
     * other in-flight submissions. Afterwards this submission's serial is recorded as the pending event of every
     * touched buffer.
     *
     * <p>{@code descriptorSet} (0 if none) is freed, and {@code staging} (if any) returned to the pool, once the
     * submission completes.
     */
    private long submit(Recorder recorder, VulkanResource[] touched, long descriptorSet, @Nullable Staging staging) {
        if (closed) throw new IllegalStateException("VulkanGpuExecutor is closed");
        try (MemoryStack stack = stackPush()) {
            Submission s = acquireSubmission(stack); // also refreshes completedSerial
            final long serial = submitSerial + 1;

            boolean needBarrier = false;
            for (VulkanResource r : touched) {
                long p = r.pendingSerial();
                // unfinished, and not already behind a barrier that a later submission opened with
                if (p > completedSerial && p >= barrierSerial) {
                    needBarrier = true;
                    break;
                }
            }

            VkCommandBufferBeginInfo begin = VkCommandBufferBeginInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
                    .flags(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
            check(vkBeginCommandBuffer(s.cb, begin), "vkBeginCommandBuffer");

            if (needBarrier) {
                // Host writes into staging need no barrier: vkQueueSubmit already makes them visible.
                VkMemoryBarrier.Buffer barrier = VkMemoryBarrier.calloc(1, stack);
                barrier.get(0)
                        .sType(VK_STRUCTURE_TYPE_MEMORY_BARRIER)
                        .srcAccessMask(VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT)
                        .dstAccessMask(VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
                                | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
                vkCmdPipelineBarrier(s.cb,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                        0, barrier, null, null);
            }

            recorder.record(s.cb, stack);
            check(vkEndCommandBuffer(s.cb), "vkEndCommandBuffer");

            VkSubmitInfo submit = VkSubmitInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_SUBMIT_INFO)
                    .pCommandBuffers(stack.pointers(s.cb));
            check(vkQueueSubmit(queue, submit, s.fence), "vkQueueSubmit");

            submitSerial = serial;
            if (needBarrier)
                barrierSerial = serial;
            s.serial = serial;
            if (descriptorSet != VK_NULL_HANDLE)
                s.descriptorSets.add(descriptorSet);
            s.staging = staging;
            inFlight.addLast(s);

            VulkanEvent event = new VulkanEvent(serial);
            for (VulkanResource r : touched)
                r.recordEvent(event);
            return serial;
        }
    }

    /**
     * Retires finished submissions, oldest first: frees their descriptor sets, recycles their staging blocks and
     * command buffers/fences, and destroys buffers that were waiting on them. Stopping at the first unfinished one
     * is what makes {@code completedSerial} a true watermark ("everything up to N is done"). A finished submission
     * that a host thread is still blocked on is counted as complete but not recycled until that thread leaves. Lock held.
     */
    private void reap() {
        while (!inFlight.isEmpty()) {
            Submission s = inFlight.peekFirst();
            int status = vkGetFenceStatus(device, s.fence);
            if (status == VK_NOT_READY)
                break;
            check(status, "vkGetFenceStatus");

            if (s.serial > completedSerial)
                completedSerial = s.serial;
            if (s.waiters > 0)
                break; // its fence can't be reset yet; the waiter calls reap() again on its way out

            inFlight.pollFirst();
            for (long set : s.descriptorSets)
                vkFreeDescriptorSets(device, descriptorPool, set);
            s.descriptorSets.clear();
            if (s.staging != null) {
                freeStaging.add(s.staging);
                s.staging = null;
            }
            idle.addLast(s);
        }

        if (!graveyard.isEmpty())
            graveyard.removeIf(dead -> {
                if (dead.serial() <= completedSerial) {
                    destroyBuffer(dead.buffer(), dead.memory());
                    return true;
                }
                return false;
            });
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
        List<VulkanResource> touched = new ArrayList<>(buffers.length);
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
                        touched.add(arg);
                    }
                    case IMMEDIATE -> {
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

            final VulkanResource[] touchedArr = touched.toArray(new VulkanResource[0]);
            final long serial;
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
                                .pBufferInfo(infos.slice(i, 1))
                                .descriptorCount(1); // set explicitly: never rely on the auto-sized count shared by three pointer members
                    }
                    vkUpdateDescriptorSets(device, writes, null);
                }

                serial = submit((cb, stack) -> {
                    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
                    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.pipelineLayout, 0, stack.longs(set), null);
                    vkCmdPushConstants(cb, kernel.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push);
                    vkCmdDispatch(cb, gx, gy, 1);
                }, touchedArr, set, null);
            }

            if (blocking) { // wait outside the lock: other threads can keep enqueuing meanwhile
                waitAndReap(serial);
                for (VulkanResource r : touchedArr)
                    r.settleThrough(serial);
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
                    //.descriptorSetCount(1);
            LongBuffer pSet = stack.mallocLong(1);

            int r = vkAllocateDescriptorSets(device, info, pSet);
            if (r != VK_SUCCESS) {
                // pool exhausted by in-flight work: let it drain (which frees their sets) and try once more
                waitAndReap(submitSerial);
                r = vkAllocateDescriptorSets(device, info, pSet);
            }
            check(r, "vkAllocateDescriptorSets");
            return pSet.get(0);
        }
    }

    // ---------------------------------------------------------------- host transfers

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

    /**
     * Host -> device. <b>Non-blocking:</b> the host bytes are copied into a pooled staging block, the device copy is
     * submitted, and this returns -- the caller's array is immediately reusable, and the upload completes in the
     * background. It is recorded as {@code dst}'s pending event, so anything that touches {@code dst} next is ordered
     * after it device-side, while work on other buffers is unaffected.
     */
    private void write(GpuResource dst, long dstByteOffset, long totalBytes, StageWriter writer) {
        if (totalBytes <= 0) return;
        VulkanResource target = res(dst);
        final VulkanResource[] touched = {target};
        synchronized (lock) {
            long pos = 0;
            while (pos < totalBytes) {
                int chunk = (int) Math.min(totalBytes - pos, MAX_STAGING_BYTES);
                Staging st = acquireStaging(chunk);
                try {
                    writer.write(view(st, chunk), pos, chunk);

                    final long dstOffset = dstByteOffset + pos;
                    final int bytes = chunk;
                    submit((cb, stack) -> {
                        VkBufferCopy.Buffer region = VkBufferCopy.calloc(1, stack).srcOffset(0).dstOffset(dstOffset).size(bytes);
                        vkCmdCopyBuffer(cb, st.buffer(), target.buffer, region);
                    }, touched, VK_NULL_HANDLE, st); // the submission now owns `st` and recycles it when done
                } catch (RuntimeException | Error e) {
                    freeStaging.add(st); // never reached the queue
                    throw e;
                }
                pos += chunk;
            }
        }
    }

    /**
     * Device -> host. Necessarily blocking, but only on its own serial (everything it depends on is ordered before it
     * device-side), and the wait happens outside the executor lock.
     */
    private void read(GpuResource src, long srcByteOffset, long totalBytes, StageReader reader) {
        if (totalBytes <= 0) return;
        VulkanResource source = res(src);
        final VulkanResource[] touched = {source};
        long pos = 0;
        while (pos < totalBytes) {
            int chunk = (int) Math.min(totalBytes - pos, MAX_STAGING_BYTES);
            final long srcOffset = srcByteOffset + pos;
            final int bytes = chunk;

            final Staging st;
            final long serial;
            synchronized (lock) {
                st = acquireStaging(chunk); // owned by this call, not by the submission: we read it after the wait
                try {
                    serial = submit((cb, stack) -> {
                        VkBufferCopy.Buffer region = VkBufferCopy.calloc(1, stack).srcOffset(srcOffset).dstOffset(0).size(bytes);
                        vkCmdCopyBuffer(cb, source.buffer, st.buffer(), region);

                        // make the transfer's writes available to the host
                        VkMemoryBarrier.Buffer toHost = VkMemoryBarrier.calloc(1, stack);
                        toHost.get(0)
                                .sType(VK_STRUCTURE_TYPE_MEMORY_BARRIER)
                                .srcAccessMask(VK_ACCESS_TRANSFER_WRITE_BIT)
                                .dstAccessMask(VK_ACCESS_HOST_READ_BIT);
                        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, toHost, null, null);
                    }, touched, VK_NULL_HANDLE, null);
                } catch (RuntimeException | Error e) {
                    freeStaging.add(st);
                    throw e;
                }
            }

            try {
                waitAndReap(serial);
                source.settleThrough(serial);
                reader.read(view(st, chunk), pos, chunk);
            } finally {
                synchronized (lock) {
                    if (!closed)
                        freeStaging.add(st);
                }
            }
            pos += chunk;
        }
    }

    // ---- doubles

    @Override
    public void uploadDoubles(GpuResource buffer, int elementOffset, double[] host, int hostOffset, int length) {
        write(buffer, (long) elementOffset * Double.BYTES, (long) length * Double.BYTES,
                (staging, pos, n) -> staging.asDoubleBuffer().put(host, hostOffset + (int) (pos / Double.BYTES), n / Double.BYTES));
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

    // ---- floats

    @Override
    public void uploadFloats(GpuResource buffer, int elementOffset, float[] host, int hostOffset, int length) {
        write(buffer, (long) elementOffset * Float.BYTES, (long) length * Float.BYTES,
                (staging, pos, n) -> staging.asFloatBuffer().put(host, hostOffset + (int) (pos / Float.BYTES), n / Float.BYTES));

    }

    @Override
    public void downloadFloats(GpuResource buffer, int elementOffset, float[] host, int length) {
        read(buffer, (long) elementOffset * Float.BYTES, (long) length * Float.BYTES,
                (staging, pos, n) -> staging.asFloatBuffer().get(host, (int) (pos / Float.BYTES), n / Float.BYTES));

    }

    @Override
    public void uploadFloatAt(GpuResource buffer, int elementOffset, float value) {
        write(buffer, (long) elementOffset * Float.BYTES, Float.BYTES, (staging, pos, n) -> staging.putFloat(0, value));

    }

    @Override
    public float downloadFloatAt(GpuResource buffer, int elementOffset) {
        float[] out = new float[1];
        read(buffer, (long) elementOffset * Float.BYTES, Float.BYTES, (staging, pos, n) -> out[0] = staging.getFloat(0));
        return out[0];
    }

    // ---- ints

    @Override
    public void uploadInts(GpuResource buffer, int elementOffset, int[] host, int hostOffset, int length) {
        write(buffer, (long) elementOffset * Integer.BYTES, (long) length * Integer.BYTES,
                (staging, pos, n) -> staging.asIntBuffer().put(host, hostOffset + (int) (pos / Integer.BYTES), n / Integer.BYTES));
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
    public void uploadBytes(GpuResource buffer, int elementOffset, byte[] host, int hostOffset, int length) {
        write(buffer, elementOffset, length, (staging, pos, n) -> staging.put(host, (int) pos+hostOffset, n));
    }

    @Override
    public void downloadBytes(GpuResource buffer, int elementOffset, byte[] host, int hostOffset, int length) {
        read(buffer, elementOffset, length, (staging, pos, n) -> staging.get(host, (int) pos+hostOffset, n));
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

    // ---------------------------------------------------------------- copy / fill (non-blocking, tracked like dispatches)

    @Override
    public void copyDoubleBuffer(GpuResource src, int srcOffset, GpuResource dst, int dstOffset, int length) {
        copy(src, (long) srcOffset * Double.BYTES, dst, (long) dstOffset * Double.BYTES, (long) length * Double.BYTES);
    }

    @Override
    public void copyFloatBuffer(GpuResource src, int srcOffset, GpuResource dst, int dstOffset, int length) {
        copy(src, (long) srcOffset * Float.BYTES, dst, (long) dstOffset * Float.BYTES, (long) length * Float.BYTES);

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
            }, new VulkanResource[]{s, d}, VK_NULL_HANDLE, null);
        }
    }

    private void fill32(GpuResource buffer, long byteOffset, long bytes, int pattern) {
        VulkanResource b = res(buffer);
        synchronized (lock) {
            submit((cb, stack) -> vkCmdFillBuffer(cb, b.buffer, byteOffset, bytes, pattern), new VulkanResource[]{b}, VK_NULL_HANDLE, null);
        }
    }

    @Override
    public void fillDoubleBuffer(GpuResource buffer, int offset, double value, int length) {
        if (length <= 0) return;
        if (fillDoubleKernel == null && !hasFloat64)
            throw new UnsupportedOperationException("Vulkan device '" + deviceName + "' has no shaderFloat64; double buffers can't be used");
        long bits = Double.doubleToRawLongBits(value);
        int lo = (int) bits;
        int hi = (int) (bits >>> 32);
        if (lo == hi) { // includes 0.0: expressible as a repeated 32-bit pattern
            //TODO WHY ? is it not possible to fill with 64 ? also it's not used afterward so why the
            fill32(buffer, (long) offset * Double.BYTES, (long) length * Double.BYTES, lo);
            return;
        }
        dispatch(fillDoubleKernel, length, 0, new GpuResource[]{buffer, scalar(offset), scalar(value)}, false);
    }

    @Override
    public void fillFloatBuffer(GpuResource buffer, int offset, float value, int length) {
        if (length <= 0) return;
        int bits = Float.floatToRawIntBits(value);
        fill32(buffer, (long) offset * Float.BYTES, (long) length * Float.BYTES, bits);
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

    /** Blocks until every previously submitted operation has completed. Waits outside the lock, so other threads keep enqueuing. */
    @Override
    public void finish() {
        long last;
        synchronized (lock) {
            if (closed) return;
            last = submitSerial;
        }
        waitAndReap(last);
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

            for (Submission s : idle)
                vkDestroyFence(device, s.fence, null);
            idle.clear();
            for (Submission s : inFlight) // only non-empty if a waiter was still pinned; the device is idle, so safe
                vkDestroyFence(device, s.fence, null);
            inFlight.clear();
            for (Staging s : freeStaging)
                destroyStaging(s);
            freeStaging.clear();

            vkDestroyDescriptorPool(device, descriptorPool, null);
            vkDestroyCommandPool(device, commandPool, null); // frees its command buffers
            memoryProperties.free();
            vkDestroyDevice(device, null);
            if (debugMessenger != VK_NULL_HANDLE)
                EXTDebugUtils.vkDestroyDebugUtilsMessengerEXT(instance, debugMessenger, null);
            vkDestroyInstance(instance, null);
            if (debugCallback != null)
                debugCallback.free();
        }
    }
}