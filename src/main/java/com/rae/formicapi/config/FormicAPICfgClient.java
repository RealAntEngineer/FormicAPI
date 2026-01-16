package com.rae.formicapi.config;


import com.rae.formicapi.FormicAPI;
import net.createmod.catnip.config.ConfigBase;
import org.jetbrains.annotations.NotNull;

public class FormicAPICfgClient extends ConfigBase {


    public final UnitConfig units = nested(0, UnitConfig::new, Comments.units);
    @Override
    public @NotNull String getName() {
        return FormicAPI.MODID +".client";
    }

    private static class Comments {
        static String units = "Units used";
    }

}
