/**
 * Backend for Gpu accelerated computation, implemented through OpenCL (which also make possible the
 * use of cpu with the same classes) and Vulkan (see the {@code vulkan} subpackage). The {@code Gpu*Vector}/{@code Gpu*Matrix}
 * classes only talk to {@link com.rae.formicapi.foundation.math.operators.backend.gpu.GpuExecutor} and declare their
 * kernels as {@link com.rae.formicapi.foundation.math.operators.backend.gpu.PortableKernel}s (one OpenCL C and one GLSL
 * source each), so they run unchanged on either backend.
 *
 */
@NonnullDefault
package com.rae.formicapi.foundation.math.operators.backend.gpu;

import org.lwjgl.system.NonnullDefault;