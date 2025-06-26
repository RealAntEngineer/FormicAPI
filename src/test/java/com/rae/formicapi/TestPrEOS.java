package com.rae.formicapi;

import com.rae.formicapi.thermal_utilities.EOSLibrary;
import com.rae.formicapi.thermal_utilities.eos.PengRobinsonEOS;
import com.rae.formicapi.thermal_utilities.helper.WaterCubicEOS;

public class TestPrEOS {
    public static void main(String[] args) {
        PengRobinsonEOS EOS = EOSLibrary.getPRWaterEOS();

        System.out.println("pressure limit : "+
                WaterCubicEOS.isentropicPressureChange2(600, 5e6f, 1f, 1e4f));
    }
}
