package com.rae.formicapi.content.config;

import com.rae.formicapi.foundation.units.Pressure;
import com.rae.formicapi.foundation.units.IrradiationFlux;
import com.rae.formicapi.foundation.units.Temperature;
import net.createmod.catnip.config.ConfigBase;
import org.jetbrains.annotations.NotNull;

public class UnitConfig extends ConfigBase {
    public final ConfigEnum<Temperature> temperature = e(Temperature.CELSIUS,"temperature", Comments.temperature);
    public final ConfigEnum<Pressure>        pressure      = e(Pressure.ATMOSPHERES,"pressure", Comments.pressure);
    public final ConfigEnum<IrradiationFlux> radiationFlux = e(IrradiationFlux.BECQUERELS,"radiation_flux", Comments.radiationFlux);


    @Override
    public @NotNull String getName() {
        return "units";
    }

    private static class Comments {
        static final String temperature = "unit used for temperature";
        static final String pressure    = "unit used for pressure";
        static final String radiationFlux = "unit used for radiation activity";


    }
}
