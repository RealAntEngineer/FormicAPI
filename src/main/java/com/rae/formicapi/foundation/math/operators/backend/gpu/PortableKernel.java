package com.rae.formicapi.foundation.math.operators.backend.gpu;

import org.jetbrains.annotations.Nullable;

import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * A kernel written once per shading language and usable on any backend that
 * has a source for it: OpenCL C for the OpenCL backend, GLSL compute for the
 * Vulkan backend. This is what lets {@code Gpu*Vector}/{@code Gpu*Matrix}
 * declare their kernels as plain {@code static final} constants and never
 * mention a concrete backend:
 *
 * <pre>
 * private static final PortableKernel AXPY = new PortableKernel("axpy")
 *         .opencl("...OpenCL C...", "cl_khr_fp64")
 *         .glsl("...GLSL...", "shaderFloat64");
 * </pre>
 *
 * <p>{@link #bind} asks the executor to build its native kernel from the
 * matching source (see {@link GpuExecutor#createKernel}); every
 * {@code use}/{@code useBlocking} afterwards forwards to that native kernel.
 * A kernel with no source for the executor's API, or whose requirements
 * (OpenCL extensions / Vulkan device features) the device doesn't meet,
 * fails at bind with {@link UnsupportedOperationException} -- check
 * {@link GpuExecutor#supports(Kernel)} first for optional kernels.
 *
 * <p><b>One active executor at a time per kernel.</b> Like
 * {@code OpenCLKernel} before it, the kernel dispatches against whichever
 * executor it was bound to last. Native kernels are cached per executor, so
 * re-binding to an executor already seen is a cheap lookup, not a recompile.
 *
 * <h3>GLSL conventions (Vulkan)</h3>
 * Write only the body of the shader -- no {@code #version}, no
 * {@code local_size}, no {@code #extension}; the Vulkan backend adds those
 * (extensions are derived from the feature names given to {@link #glsl}).
 * <ul>
 *   <li>Buffer arguments map, in order, to {@code binding = 0, 1, 2...}
 *       (scalar and local-memory arguments don't consume a binding).</li>
 *   <li>Scalar arguments are read from the push-constant block, whose first
 *       member is always an implicit {@code int gsz} (the dispatch's
 *       {@code globalSize}), followed by the scalars in argument order.</li>
 *   <li>Use {@code GID} instead of {@code gl_GlobalInvocationID.x} and
 *       {@code GROUP_ID} instead of {@code gl_WorkGroupID.x} (they transparently
 *       support dispatches wider than the device's group-count limit), and
 *       guard with {@code if (GID >= uint(pc.gsz)) return;} since Vulkan
 *       rounds the dispatch up to whole work-groups.</li>
 *   <li>Local-memory arguments are ignored; declare
 *       {@code shared T s[gl_WorkGroupSize.x];} instead.</li>
 * </ul>
 */
public final class PortableKernel implements Kernel {

    private static final Set<PortableKernel> ALL = ConcurrentHashMap.newKeySet();

    private final String name;

    private @Nullable String openclSource;
    private String[]         openclRequirements = new String[0];
    private @Nullable String glslSource;
    private String[]         glslRequirements   = new String[0];

    private final Map<GpuExecutor, Kernel> perExecutor = new ConcurrentHashMap<>();
    private volatile @Nullable Kernel      current;

    public PortableKernel(String name) {
        this.name = name;
        ALL.add(this);
    }

    /** OpenCL C source containing a {@code __kernel} named like this kernel; {@code extensions} are required device extensions, e.g. {@code "cl_khr_fp64"}. */
    public PortableKernel opencl(String source, String... extensions) {
        this.openclSource = source;
        this.openclRequirements = extensions;
        return this;
    }

    /** GLSL compute shader body (see class doc); {@code features} are required Vulkan device features, e.g. {@code "shaderFloat64"}, {@code "shaderInt64"}, {@code "shaderBufferInt64Atomics"}, {@code "storageBuffer8BitAccess"}. */
    public PortableKernel glsl(String source, String... features) {
        this.glslSource = source;
        this.glslRequirements = features;
        return this;
    }

    public String name() {
        return name;
    }

    public @Nullable String openclSource() {
        return openclSource;
    }

    public String[] openclRequirements() {
        return openclRequirements;
    }

    public @Nullable String glslSource() {
        return glslSource;
    }

    public String[] glslRequirements() {
        return glslRequirements;
    }

    @Override
    public synchronized Kernel bind(GpuExecutor context) {
        Kernel native_ = perExecutor.get(context);
        if (native_ == null) {
            native_ = context.createKernel(this);
            perExecutor.put(context, native_);
        }
        native_.bind(context);
        current = native_;
        return this;
    }

    private Kernel requireCurrent() {
        Kernel k = current;
        if (k == null)
            throw new IllegalStateException("Kernel '" + name + "' was never bind()ed -- call bind(executor) before use(...)");
        return k;
    }

    @Override
    public void use(int globalSize, int localSize, GpuResource... args) {
        requireCurrent().use(globalSize, localSize, args);
    }

    @Override
    public void useBlocking(int globalSize, int localSize, GpuResource... args) {
        requireCurrent().useBlocking(globalSize, localSize, args);
    }

    @Override
    public int maxLocalSize() {
        return requireCurrent().maxLocalSize();
    }

    /** Called by an executor when it closes: drops every cached native kernel built for it. */
    public static void forgetExecutor(GpuExecutor executor) {
        for (PortableKernel k : ALL) {
            synchronized (k) {
                Kernel removed = k.perExecutor.remove(executor);
                if (removed != null && k.current == removed)
                    k.current = null;
            }
        }
    }
}
