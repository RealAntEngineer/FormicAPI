package com.rae.formicapi.foundation.math.pde;

public enum SymbolRole {

    /**
     * Variable being solved (x).
     */
    UNKNOWN,

    /**
     * Used to compute entries of A.
     */
    COEFFICIENT,

    /**
     * Runtime value contributing to b.
     * Typically another field or the previous
     * nonlinear iteration.
     */
    EVALUATED_FIELD,

    /**
     * Compile-time constant.
     */
    CONSTANT
}
