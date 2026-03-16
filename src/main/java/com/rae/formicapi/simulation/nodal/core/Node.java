package com.rae.formicapi.simulation.nodal.core;

public interface Node {

    boolean isUnknown();

    int getId();

    void setId(int id);

    double getValue();   // known value if fixed

    void setValue(double value);
}
