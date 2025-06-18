package com.rae.formicapi.thermal_utilities;

import net.createmod.catnip.data.Couple;
import oshi.util.tuples.Pair;

import java.util.ArrayList;
import java.util.List;

public abstract class CubicEOS implements EquationOfState{
    @Override
    public abstract double pressure(double temperature, double volumeMolar);


    abstract Couple<Double> findSpinodalPoints(double T);
    abstract double[] getZFactors(double T, double P);


}
