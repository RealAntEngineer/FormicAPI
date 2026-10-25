package com.rae.formicapi.foundation.math.operators.backend.gpu.vulkan;

import com.rae.formicapi.foundation.math.operators.backend.gpu.GpuExecutor;
import com.rae.formicapi.foundation.math.operators.backend.gpu.GpuResource;
import com.rae.formicapi.foundation.math.operators.backend.gpu.Kernel;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.*;

import java.nio.ByteBuffer;
import java.nio.LongBuffer;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.util.shaderc.Shaderc.*;
import static org.lwjgl.vulkan.VK10.*;

/**
 * One Vulkan compute kernel: a GLSL compute shader (see
 * {@link com.rae.formicapi.foundation.math.operators.backend.gpu.PortableKernel}
 * for the conventions the source follows) compiled to SPIR-V through shaderc,
 * wrapped in a shader module, descriptor-set layout and pipeline layout.
 *
 * <p><b>Work-group size is a specialization constant.</b> The shader declares
 * {@code local_size_x_id = 0}, and one {@code VkPipeline} is built lazily per
 * distinct local size actually requested, then cached -- that's how reduction
 * kernels get exactly the group size their scratch/partials buffers were
 * sized for, while plain elementwise kernels all share the executor's default.
 *
 * <p>Instances are created by {@link VulkanGpuExecutor#createKernel} and are
 * bound to that executor for life; they're destroyed when it closes.
 */
public final class VulkanKernel implements Kernel {

    /** Push constants are limited to 128 bytes on every conforming device: the implicit {@code gsz} plus up to 124 bytes of scalars. */
    static final        int PUSH_BYTES   = 128;
    public static final int MAX_BINDINGS = 8;

    private static final Pattern BINDING = Pattern.compile("binding\\s*=\\s*(\\d+)");

    private final VulkanGpuExecutor executor;
    private final String name;

    final int  bindingCount;
    final long descriptorSetLayout;
    final long pipelineLayout;
    private final long shaderModule;

    private final Map<Integer, Long> pipelines = new HashMap<>();
    private boolean destroyed;

    VulkanKernel(VulkanGpuExecutor executor, String name, String glslBody, String[] features) {
        this.executor = executor;
        this.name = name;

        int max = -1;
        Matcher m = BINDING.matcher(glslBody);
        while (m.find())
            max = Math.max(max, Integer.parseInt(m.group(1)));
        this.bindingCount = max + 1;
        if (bindingCount < 1 || bindingCount > MAX_BINDINGS)
            throw new IllegalArgumentException("Kernel '" + name + "' declares " + bindingCount + " buffer bindings; supported range is 1.." + MAX_BINDINGS);

        VkDevice device = executor.device();
        try (MemoryStack stack = stackPush()) {
            this.shaderModule = createShaderModule(device, stack, compile(name, wrap(glslBody, features)));

            VkDescriptorSetLayoutBinding.Buffer bindings = VkDescriptorSetLayoutBinding.calloc(bindingCount, stack);
            for (int i = 0; i < bindingCount; i++)
                bindings.get(i)
                        .binding(i)
                        .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                        .descriptorCount(1)
                        .stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);

            VkDescriptorSetLayoutCreateInfo setInfo = VkDescriptorSetLayoutCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO)
                    .pBindings(bindings);
            LongBuffer pSetLayout = stack.mallocLong(1);
            VulkanGpuExecutor.check(vkCreateDescriptorSetLayout(device, setInfo, null, pSetLayout), "vkCreateDescriptorSetLayout");
            this.descriptorSetLayout = pSetLayout.get(0);

            VkPushConstantRange.Buffer pushRange = VkPushConstantRange.calloc(1, stack)
                    .stageFlags(VK_SHADER_STAGE_COMPUTE_BIT)
                    .offset(0)
                    .size(PUSH_BYTES);
            VkPipelineLayoutCreateInfo layoutInfo = VkPipelineLayoutCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO)
                    .pSetLayouts(stack.longs(descriptorSetLayout))
                    .pPushConstantRanges(pushRange);
            LongBuffer pLayout = stack.mallocLong(1);
            VulkanGpuExecutor.check(vkCreatePipelineLayout(device, layoutInfo, null, pLayout), "vkCreatePipelineLayout");
            this.pipelineLayout = pLayout.get(0);
        }
    }

    // ---------------------------------------------------------------- GLSL assembly / compilation

    /**
     * Wraps a kernel body with everything the Vulkan backend owns: version,
     * the {@code #extension} lines implied by the requested device features,
     * the specialization-constant work-group size, and the {@code GID} /
     * {@code GROUP_ID} helpers (which fold a 2D dispatch back into one linear
     * index so kernels still work past the device's per-dimension group-count limit).
     */
    public static String wrap(String body, String[] features) {
        StringBuilder sb = new StringBuilder("#version 450\n");
        for (String feature : features) {
            switch (feature) {
                case "shaderFloat64" -> { /* double is core in GLSL 4.50; the device feature is checked at createKernel */ }
                case "shaderInt64" -> sb.append("#extension GL_ARB_gpu_shader_int64 : require\n");
                case "shaderBufferInt64Atomics" -> sb.append("#extension GL_EXT_shader_atomic_int64 : require\n");
                case "storageBuffer8BitAccess" -> sb.append("#extension GL_EXT_shader_8bit_storage : require\n");
                case "shaderInt8" -> sb.append("#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require\n");
                default -> throw new IllegalArgumentException("Unknown Vulkan feature requirement: " + feature);
            }
        }
        sb.append("layout(local_size_x_id = 0) in;\n");
        sb.append("#define GID (gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * gl_NumWorkGroups.x * gl_WorkGroupSize.x)\n");
        sb.append("#define GROUP_ID (gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x)\n");
        sb.append(body);
        return sb.toString();
    }

    public static ByteBuffer compile(String name, String source) {
        long compiler = shaderc_compiler_initialize();
        long options = shaderc_compile_options_initialize();
        if (compiler == 0L || options == 0L)
            throw new IllegalStateException("Unable to initialise shaderc");
        try {
            shaderc_compile_options_set_target_env(options, shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
            shaderc_compile_options_set_optimization_level(options, shaderc_optimization_level_performance);

            long result = shaderc_compile_into_spv(compiler, source, shaderc_glsl_compute_shader, name + ".comp", "main", options);
            try {
                if (shaderc_result_get_compilation_status(result) != shaderc_compilation_status_success)
                    throw new RuntimeException("Vulkan kernel '" + name + "' failed to compile:\n" + shaderc_result_get_error_message(result));

                ByteBuffer spirv = shaderc_result_get_bytes(result);
                if (spirv == null)
                    throw new RuntimeException("shaderc returned no SPIR-V for kernel '" + name + "'");

                // the result owns this memory; copy it out before releasing the result
                ByteBuffer copy = org.lwjgl.system.MemoryUtil.memAlloc(spirv.remaining());
                copy.put(spirv).flip();
                return copy;
            } finally {
                shaderc_result_release(result);
            }
        } finally {
            shaderc_compile_options_release(options);
            shaderc_compiler_release(compiler);
        }
    }

    private static long createShaderModule(VkDevice device, MemoryStack stack, ByteBuffer spirv) {
        try {
            VkShaderModuleCreateInfo info = VkShaderModuleCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO)
                    .pCode(spirv);
            LongBuffer pModule = stack.mallocLong(1);
            VulkanGpuExecutor.check(vkCreateShaderModule(device, info, null, pModule), "vkCreateShaderModule");
            return pModule.get(0);
        } finally {
            org.lwjgl.system.MemoryUtil.memFree(spirv);
        }
    }

    // ---------------------------------------------------------------- pipelines

    /** The pipeline specialized to work-group size {@code localSize}, building it on first request. */
    synchronized long pipeline(int localSize) {
        Long cached = pipelines.get(localSize);
        if (cached != null)
            return cached;

        VkDevice device = executor.device();
        try (MemoryStack stack = stackPush()) {
            VkSpecializationMapEntry.Buffer entries = VkSpecializationMapEntry.calloc(1, stack);
            entries.get(0).constantID(0).offset(0).size(Integer.BYTES);

            ByteBuffer data = stack.malloc(Integer.BYTES);
            data.putInt(0, localSize);

            VkSpecializationInfo spec = VkSpecializationInfo.calloc(stack)
                    .pMapEntries(entries)
                    .pData(data);

            VkComputePipelineCreateInfo.Buffer info = VkComputePipelineCreateInfo.calloc(1, stack);
            info.get(0)
                    .sType(VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO)
                    .layout(pipelineLayout);
            info.get(0).stage()
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO)
                    .stage(VK_SHADER_STAGE_COMPUTE_BIT)
                    .module(shaderModule)
                    .pName(stack.UTF8("main"))
                    .pSpecializationInfo(spec);

            LongBuffer pPipeline = stack.mallocLong(1);
            VulkanGpuExecutor.check(vkCreateComputePipelines(device, VK_NULL_HANDLE, info, null, pPipeline),
                    "vkCreateComputePipelines(" + name + ", local=" + localSize + ")");
            pipelines.put(localSize, pPipeline.get(0));
            return pPipeline.get(0);
        }
    }

    // ---------------------------------------------------------------- Kernel

    @Override
    public Kernel bind(GpuExecutor context) {
        if (context != executor)
            throw new IllegalArgumentException("Vulkan kernel '" + name + "' belongs to a different executor");
        return this;
    }

    @Override
    public void use(int globalSize, int localSize, GpuResource... args) {
        executor.dispatch(this, globalSize, localSize, args, false);
    }

    @Override
    public void useBlocking(int globalSize, int localSize, GpuResource... args) {
        executor.dispatch(this, globalSize, localSize, args, true);
    }

    @Override
    public int maxLocalSize() {
        return (int) executor.maxWorkGroupSize();
    }

    String name() {
        return name;
    }

    /** Called from {@link VulkanGpuExecutor#close()} after the queue is idle. */
    synchronized void destroy() {
        if (destroyed) return;
        destroyed = true;
        VkDevice device = executor.device();
        for (long p : pipelines.values())
            vkDestroyPipeline(device, p, null);
        pipelines.clear();
        vkDestroyPipelineLayout(device, pipelineLayout, null);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, null);
        vkDestroyShaderModule(device, shaderModule, null);
    }
}
