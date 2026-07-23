package com.rae.formicapi.foundation.math.operators.physics;

/**
 * Maps a (cell, variable) pair to a slot in the global mixed-variable state vector.
 *
 * <p>Implementations own the interleaving of variables (e.g. all variables of a cell
 * stored contiguously). Matrix classes never compute this offset themselves — they ask
 * the layout, so the physics stays out of the solver/matrix code entirely.
 */
public interface VariableLayout {

    /** Number of variables stored per cell (e.g. 5 for u, v, w, P, T). */
    int variablesPerCell();

    /** Global index of {@code variable} at {@code cell} in the state vector. */
    int index(int cell, Variable variable);
}