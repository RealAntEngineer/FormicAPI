/**
 * everything in that package had a heavy use of ClaudeIA and/or ChatGPT treat design and assomption with a grain of salt
 */
//TODO let's make the walls only if the there is not enough resources and/or the executable is already
// receiving work (so you need it to finish first), that will allow real parallel work to not block each others.
//TODO CPU also need blocking before
@ParametersAreNonnullByDefault
@NonnullDefault
package com.rae.formicapi.foundation.math.operators.backend;

import org.lwjgl.system.NonnullDefault;

import javax.annotation.ParametersAreNonnullByDefault;