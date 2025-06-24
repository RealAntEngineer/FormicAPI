package io.github.realantengineer.formicapi;

import io.github.realantengineer.formicapi.thermal_utilities.EOSLibrary;
import io.github.realantengineer.formicapi.thermal_utilities.eos.PengRobinsonEOS;
import io.github.realantengineer.formicapi.thermal_utilities.helper.WaterCubicEOS;

public class TestPrEOS {
    public static void main(String[] args) {
        PengRobinsonEOS EOS = EOSLibrary.getPRWaterEOS();

        System.out.println("pressure limit : "+
                WaterCubicEOS.isentropicPressureChange2(600, 5e6f, 1f, 1e4f));
    }
}
