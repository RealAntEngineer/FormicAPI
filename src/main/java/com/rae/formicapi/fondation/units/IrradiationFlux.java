package com.rae.formicapi.fondation.units;

import com.rae.formicapi.FormicApiLang;
import net.minecraft.network.chat.Component;

public enum IrradiationFlux implements IUnit {
    // Assuming base unit is MBq (Becquerels)
    BECQUERELS(1e6f),
    CURIES(1 / (37e9f)),         // 1 Ci = 37 GBq = 37_000 MBq
    // Approximations for absorbed dose rate assuming 1 MeV per decay and full absorption in 1 kg
    GRAY_PER_SECOND(1.602e-7f / 1e6f),   // ~1 MeV per decay
    RAD_PER_SECOND(1.602e-5f / 1e6f),   // 1 Gy = 100 rad
    SIEVERT_PER_SECOND(1.602e-7f / 1e6f); // Q factor assumed 1

    final Component symbol;
    private final float a;

    IrradiationFlux(float a) {
        this.a = a;
        this.symbol = FormicApiLang.translate("units.radiation_flux.symbol." + name().toLowerCase()).component();

    }

    public float convert(float radiationFluxInMBq) {
        return radiationFluxInMBq * a;
    }

    public Component getSymbol() {
        return symbol;
    }
}
