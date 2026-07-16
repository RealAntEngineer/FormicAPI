package com.rae.formicapi.foundation.math.pde;

public enum SymbolRole {

    /**
     * Variable being solved (x).
     */
    UNKNOWN(true, true),

    /**
     * Runtime value contributing to b.
     * Typically, another field like velocity in the context of temperature transport
     */
    EVALUATED_FIELD(false, true),

    /**
     * Used to compute entries of A, it supposed to be constant across time and space
     */
    COEFFICIENT,

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

    public boolean isTimeDifferentiable() {
        return timeDifferentiable;
    }

    public boolean isSpaceDifferentiable() {
        return spaceDifferentiable;
    }
}
