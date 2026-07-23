package com.rae.formicapi.foundation.math.operators.physics;

/**
 * Standard cell-centered layout: all variables of a cell are stored contiguously.
 *
 * <pre>
 * cell 0: u0 v0 w0 P0 T0
 * cell 1: u1 v1 w1 P1 T1
 * ...
 * </pre>
 */
public final class CellCenteredVariableLayout implements VariableLayout {

    @Override
    public int variablesPerCell() {
        return Variable.values().length;
    }

    @Override
    public int index(int cell, Variable variable) {
        return cell * variablesPerCell() + variable.ordinal();
    }
}