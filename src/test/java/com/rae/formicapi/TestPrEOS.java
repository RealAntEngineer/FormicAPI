package com.rae.formicapi;

import com.rae.formicapi.thermal_utilities.EOSLibrary;
import com.rae.formicapi.thermal_utilities.PengRobinsonEOS;
import net.createmod.catnip.data.Couple;

public class TestPrEOS {
    public static void main(String[] args) {
        PengRobinsonEOS EOS = EOSLibrary.getPRWaterEOS();

        double T = 500;
        Couple<Double> V = EOS.getSaturationVolumes(T);

        double s_liq = EOS.totalEntropy(T, V.get(true));
        double s_vap = EOS.totalEntropy(T, V.get(false));

        System.out.println("s_liq = " + s_liq);
        System.out.println("s_vap = " + s_vap);


        for (T = 0 ; T < 1000 ; T++) {
            System.out.println(EOS.getSaturationVolumes(T));
        }
    }
}
