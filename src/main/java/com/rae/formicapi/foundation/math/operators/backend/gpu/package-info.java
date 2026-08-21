/**
 * Backend for Gpu accelerated computation, right now is implemented through OpenCL (which also make possible the
 * use of cpu with the same classes) and will be ported to vulkan in the future
 *
 */
@ParametersAreNonnullByDefault
@NonnullDefault
package com.rae.formicapi.foundation.math.operators.backend.gpu;

import org.lwjgl.system.NonnullDefault;

import javax.annotation.ParametersAreNonnullByDefault;