package com.rae.formicapi;

import com.rae.formicapi.units.Pressure;
import com.rae.formicapi.units.RadiationFlux;
import com.rae.formicapi.units.Temperature;
import com.rae.formicapi.config.FormicAPIConfigs;
import com.simibubi.create.foundation.utility.CreateLang;
import net.createmod.catnip.lang.Lang;
import net.createmod.catnip.lang.LangBuilder;
import net.createmod.catnip.lang.LangNumberFormat;
import net.minecraft.network.chat.Component;

import java.util.Map;
import java.util.TreeMap;

public class FormicApiLang extends Lang {
    //blatant copy of CreateLang
    static TreeMap<Double, String> MULTIPLE_SYMBOLS = new TreeMap<>();
    static {
        MULTIPLE_SYMBOLS.put(1e18, "E");
        MULTIPLE_SYMBOLS.put(1e15, "P");
        MULTIPLE_SYMBOLS.put(1e12, "T");
        MULTIPLE_SYMBOLS.put(1e9,  "G");
        MULTIPLE_SYMBOLS.put(1e6,  "M");
        MULTIPLE_SYMBOLS.put(1e3,  "k");
        MULTIPLE_SYMBOLS.put(1.0,  "");
        MULTIPLE_SYMBOLS.put(1e-3, "m");
        MULTIPLE_SYMBOLS.put(1e-6, "µ");
        MULTIPLE_SYMBOLS.put(1e-9, "n");
        MULTIPLE_SYMBOLS.put(1e-12,"p");
        MULTIPLE_SYMBOLS.put(1e-15,"f");
        MULTIPLE_SYMBOLS.put(1e-18,"a");
    }

//

    public static LangBuilder builder() {
        return new LangBuilder(FormicAPI.MODID);
    }

    public static LangBuilder numberWithSymbol(double d) {
        Map.Entry<Double, String> entry = MULTIPLE_SYMBOLS.floorEntry(d);
        if (entry == null) {
            builder().text(LangNumberFormat.format(d)+ " ");
        }
        return builder().text(LangNumberFormat.format(d/entry.getKey())+" "+entry.getValue());
    }

    public static LangBuilder translate(String langKey, Object... args) {
        return builder().translate(langKey, args);
    }

    public static LangBuilder formatTemperature(float temperature) {
        Temperature unit = FormicAPIConfigs.CLIENT.units.temperature.get();
        return CreateLang.builder().add(Component.literal("T = "))
                .text(LangNumberFormat.format(unit.convert(temperature))+ " ")
                .add(unit.getSymbol());
    }
    public static LangBuilder formatPressure(float pressure) {
        Pressure unit = FormicAPIConfigs.CLIENT.units.pressure.get();
        return CreateLang.builder().add(Component.literal("P = "))
                .add(numberWithSymbol(unit.convert(pressure)))
                .add(unit.getSymbol());
    }
    public static LangBuilder formatRadiationFlux(float radiationFlux) {
        RadiationFlux unit = FormicAPIConfigs.CLIENT.units.radiationFlux.get();
        return CreateLang.builder().add(Component.literal("activity : "))
                .add(numberWithSymbol(unit.convert(radiationFlux)))
                .add(unit.getSymbol());
    }

}
