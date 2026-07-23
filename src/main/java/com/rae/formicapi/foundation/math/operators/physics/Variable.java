package com.rae.formicapi.foundation.math.operators.physics;
 
/**
 * The physical unknowns tracked per cell in the mixed-variable state vector.
 *
 * <p>Ordinal order defines the in-cell layout used by {@link CellCenteredVariableLayout}:
 * u, v, w, P, T map to offsets 0..4 within a cell's block of the global state vector.
 */
public enum Variable {
    U,
    V,
    W,
    PRESSURE,
    TEMPERATURE;
 
    /** Bit position of this variable within a {@link VariableMask}. */
    public int bit() {
        return 1 << ordinal();
    }
}