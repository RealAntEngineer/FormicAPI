package com.rae.formicapi;

import com.rae.formicapi.thermal_utilities.EOSLibrary;
import com.rae.formicapi.thermal_utilities.PengRobinsonEOS;
import com.rae.formicapi.thermal_utilities.WaterCubicEOSTransformationHelper;
import net.createmod.catnip.data.Couple;

public class TestPrEOS {
    public static void main(String[] args) {
        PengRobinsonEOS EOS = EOSLibrary.getPRWaterEOS();

        System.out.println("pressure limit : "+
                WaterCubicEOSTransformationHelper.isentropicPressureChange2(600, 5e6f, 1f, 1e4f));
    }
}
