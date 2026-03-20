package com.rae.formicapi.fondation.simulation.nodal.core;

import com.rae.formicapi.fondation.math.operators.DenseMatrix;
import com.rae.formicapi.fondation.math.operators.DynamicCSRMatrix;
import com.rae.formicapi.fondation.math.operators.MutableMatrix;

public class SimulationContext {

    public final MutableMatrix matrix;   // the assembly matrix
    public final double[] rhs;           // right-hand side

    public SimulationContext(int numNodes, boolean useDense) {
        if (useDense) {
            matrix = new DenseMatrix(numNodes, numNodes);
        } else {
            matrix = new DynamicCSRMatrix(numNodes, numNodes);
        }
        rhs = new double[numNodes];
    }
}