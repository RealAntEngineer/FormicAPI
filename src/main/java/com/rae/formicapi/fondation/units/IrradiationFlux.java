package com.rae.formicapi.fondation.units;

import com.rae.formicapi.FormicApiLang;
import net.minecraft.network.chat.Component;

public enum IrradiationFlux implements IUnit {
    // Assuming base unit is Bq (Becquerels)
    BECQUERELS(1e6f),
    CURIES(1 / (37e9f)),         // 1 Ci = 37 GBq = 37_000 MBq
    // Approximations for absorbed dose rate assuming 1 MeV per decay and full absorption in 1 kg
    GRAY_PER_SECOND(1.602e-7f / 1e6f),   // ~1 MeV per decay
    RAD_PER_SECOND(1.602e-5f / 1e6f),   // 1 Gy = 100 rad
    SIEVERT_PER_SECOND(1.602e-7f / 1e6f); // Q factor assumed 1

    private final float a;

    IrradiationFlux(float a) {
        this.a = a;

    }

    public float convert(float radiationFluxInMBq) {
        return radiationFluxInMBq * a;
    }

}