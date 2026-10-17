package com.rae.formicapi.content.thermal_utilities;

import net.neoforged.neoforge.fluids.FluidType;

public class MutableFluidType extends FluidType {

    EOS eos;
    public MutableFluidType(Properties properties, EOS eos) {
        super(properties);
        this.eos = eos;
    }


}
