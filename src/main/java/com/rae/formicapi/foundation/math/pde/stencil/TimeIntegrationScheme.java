package com.rae.formicapi.foundation.math.pde.stencil;

/**
 * Selects the time level at which a spatial operator is evaluated in a
 * theta-method time discretization, controlling the blend between the
 * previous and current time level's contribution.
 *
 * <p>{@code theta = 0} is fully explicit (forward Euler), {@code theta = 1}
 * is fully implicit (backward Euler), {@code theta = 0.5} is Crank-Nicolson.
 *
 * <pre>
 * (U^{n+1} - U^n) / dt = theta * F(U^{n+1}) + (1 - theta) * F(U^n)
 * </pre>
 */
public enum TimeIntegrationScheme {

    EXPLICIT {
        @Override
        public double theta() {
            return 0.0;
        }
    },
    IMPLICIT {
        @Override
        public double theta() {
            return 1.0;
        }
    },
    CRANK_NICOLSON {
        @Override
        public double theta() {
            return 0.5;
        }
    };

    public abstract double theta();
}
