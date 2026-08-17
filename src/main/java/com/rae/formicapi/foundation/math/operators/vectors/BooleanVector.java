package com.rae.formicapi.foundation.math.operators.vectors;

public interface BooleanVector extends Vector {

    void set(boolean value, int idx);

    boolean get(int idx);
}
