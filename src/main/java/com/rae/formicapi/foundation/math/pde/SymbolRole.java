package com.rae.formicapi.foundation.math.pde;

public enum SymbolRole {

    /**
     * Variable being solved (x).
     */
    UNKNOWN(true, true),

    /**
     * Used to compute entries of A.
     */
    COEFFICIENT,

    /**
     * Runtime value contributing to b.
     * Typically another field or the previous
     * nonlinear iteration.
     */
    EVALUATED_FIELD(true),

    /**
     * Compile-time constant.
     */
    CONSTANT;

    final boolean timeDifferentiable;
    final boolean spaceDifferentiable;

    SymbolRole() {
        this(false, false);
    }

    SymbolRole(boolean timeDif, boolean spaceDif) {
        this.timeDifferentiable = timeDif;
        this.spaceDifferentiable = spaceDif;
    }

    SymbolRole(boolean spaceDif) {
        this(false, spaceDif);
    }

    public boolean isTimeDifferentiable() {
        return timeDifferentiable;
    }

    public boolean isSpaceDifferentiable() {
        return spaceDifferentiable;
    }
}
