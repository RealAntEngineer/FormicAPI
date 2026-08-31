package com.rae.formicapi.math_tests.gpu;

import com.rae.formicapi.foundation.math.operators.backend.gpu.GpuExecutor;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeEach;

/**
 * Shared setup for every GPU backend test class: opens a {@link GpuExecutor}
 * before each test and closes it after, skipping (not failing) the test on
 * machines without a real fp64-capable OpenCL device - most CI runners,
 * most laptops without a discrete GPU or vendor OpenCL runtime installed.
 *
 * <p>Not itself a test class ({@code @Test}-less, package-visible would be
 * fine within one package, but subclasses live in
 * {@code math_tests.vectors}/{@code math_tests.matrix}/{@code math_tests.nonlinear}
 * to mirror the main source layout, hence {@code public}/{@code protected}
 * here rather than package-private).
 */
public abstract class GpuTestSupport {

    protected static final double EPSILON = 1e-9;

    protected GpuExecutor executor;

    @BeforeEach
    protected void setUpExecutor() {
        try {
            executor = new GpuExecutor();
        } catch (RuntimeException e) {
            executor = null;
            Assumptions.assumeTrue(false, "Skipping GPU tests: no fp64-capable OpenCL device available (" + e.getMessage() + ")");
        }
    }

    @AfterEach
    protected void tearDownExecutor() {
        if (executor != null)
            executor.close();
    }
}