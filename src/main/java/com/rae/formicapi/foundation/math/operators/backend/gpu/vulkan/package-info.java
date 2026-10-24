/**
 * Vulkan implementation of the GPU backend: {@link com.rae.formicapi.foundation.math.operators.backend.gpu.vulkan.VulkanGpuExecutor}
 * (instance / device / queue, buffers, transfers), {@link com.rae.formicapi.foundation.math.operators.backend.gpu.vulkan.VulkanKernel}
 * (GLSL compute shaders compiled to SPIR-V at runtime through shaderc) and
 * {@link com.rae.formicapi.foundation.math.operators.backend.gpu.vulkan.VulkanResource}.
 *
 * <p>Built on LWJGL ({@code lwjgl-vulkan}, {@code lwjgl-shaderc}); needs a
 * Vulkan 1.2 device with {@code shaderFloat64}. Select it with
 * {@code APIChoice.setAPI(ImplementedAPI.Vulkan)}.
 */
@ParametersAreNonnullByDefault
@NonnullDefault
package com.rae.formicapi.foundation.math.operators.backend.gpu.vulkan;

import org.lwjgl.system.NonnullDefault;

import javax.annotation.ParametersAreNonnullByDefault;
