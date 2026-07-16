package com.rae.formicapi.content.config;


import com.rae.formicapi.FormicAPI;
import net.createmod.catnip.config.ConfigBase;
import org.jetbrains.annotations.NotNull;

public class FormicAPICfgCommon extends ConfigBase {


    public final UnitConfig units = nested(0, UnitConfig::new, Comments.units);

    @Override
    public @NotNull String getName() {
        return FormicAPI.MODID + ".common";
    }

    private static class Comments {
        static final String units = "Units used";
    }

}
