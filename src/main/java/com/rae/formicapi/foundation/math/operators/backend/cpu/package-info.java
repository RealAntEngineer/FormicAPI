/**
 * backend for pure cpu vectors and matrix support both parallel execution through a thread poll or pure serial
 * think of it has a fallback in case there is no gpu
 */
@ParametersAreNonnullByDefault
@NonnullDefault
package com.rae.formicapi.foundation.math.operators.backend.cpu;

import org.lwjgl.system.NonnullDefault;

import javax.annotation.ParametersAreNonnullByDefault;