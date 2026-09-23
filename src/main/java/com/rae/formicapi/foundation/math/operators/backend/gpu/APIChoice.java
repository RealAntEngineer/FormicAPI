package com.rae.formicapi.foundation.math.operators.backend.gpu;

import com.rae.formicapi.foundation.math.operators.backend.gpu.opencl.OpenCLGpuExecutor;
import org.jetbrains.annotations.Nullable;

/**
 * Chooses which backend implementation {@code Gpu*Vector}/{@code Gpu*Matrix}
 * classes run against. Set once (typically at startup) via {@link #setAPI};
 * from then on, {@link #createExecutor()} is the only place in the codebase
 * allowed to name a concrete executor class ({@link OpenCLGpuExecutor}
 * today, a future {@code VulkanGpuExecutor} tomorrow) -- everything else,
 * including every {@code Gpu*Vector}/{@code Gpu*Matrix} constructor, takes
 * and holds the backend-neutral {@link GpuExecutor} interface.
 */
public final class APIChoice {

    private static @Nullable ImplementedAPI MODE = null;

    private APIChoice() {
    }

    /**
     * Chooses the API {@link #createExecutor()} will build against. Must
     * be called once before the first {@link #createExecutor()} call --
     * there's deliberately no default, since silently picking an API for
     * someone isn't this class's call to make.
     */
    public static synchronized void setAPI(ImplementedAPI api) {
        MODE = api;
    }

    public static synchronized @Nullable ImplementedAPI currentAPI() {
        return MODE;
    }

    /**
     * Creates a new executor for whichever API {@link #setAPI} selected,
     * using that backend's own default-device selection. For OpenCL-
     * specific device selection (a particular {@code cl_device_id} rather
     * than the default), construct {@link OpenCLGpuExecutor} directly --
     * that's an intentional escape hatch, not something this generic
     * factory needs to expose.
     *
     * @throws IllegalStateException if {@link #setAPI} was never called
     */
    public static synchronized GpuExecutor createExecutor() {
        if (MODE == null)
            throw new IllegalStateException("No GPU API selected -- call APIChoice.setAPI(...) first");

        return switch (MODE) {
            case OpenCL -> new OpenCLGpuExecutor();
        };
    }

    public enum ImplementedAPI {
        OpenCL
    }
}