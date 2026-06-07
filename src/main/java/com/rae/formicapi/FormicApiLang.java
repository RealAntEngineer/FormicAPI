package com.rae.formicapi;

import com.rae.formicapi.fondation.units.IUnit;
import com.rae.formicapi.fondation.units.Pressure;
import com.rae.formicapi.fondation.units.IrradiationFlux;
import com.rae.formicapi.fondation.units.Temperature;
import com.rae.formicapi.content.config.FormicAPIConfigs;
import com.simibubi.create.foundation.utility.CreateLang;
import net.createmod.catnip.lang.Lang;
import net.createmod.catnip.lang.LangBuilder;
import net.createmod.catnip.lang.LangNumberFormat;
import net.minecraft.network.chat.Component;

import java.util.Map;
import java.util.TreeMap;

public class FormicApiLang extends Lang {
    static TreeMap<Double, String> MULTIPLE_SYMBOLS = new TreeMap<>();

    static {
        MULTIPLE_SYMBOLS.put(1e30,  "Q");  // quetta (2022)
        MULTIPLE_SYMBOLS.put(1e27,  "R");  // ronna  (2022)
        MULTIPLE_SYMBOLS.put(1e24,  "Y");  // yotta  (1991)
        MULTIPLE_SYMBOLS.put(1e21,  "Z");  // zetta  (1991)
        MULTIPLE_SYMBOLS.put(1e18,  "E");  // exa
        MULTIPLE_SYMBOLS.put(1e15,  "P");  // peta
        MULTIPLE_SYMBOLS.put(1e12,  "T");  // tera
        MULTIPLE_SYMBOLS.put(1e9,   "G");  // giga
        MULTIPLE_SYMBOLS.put(1e6,   "M");  // mega
        MULTIPLE_SYMBOLS.put(1e3,   "k");  // kilo
        MULTIPLE_SYMBOLS.put(1e2,   "h");  // hecto
        MULTIPLE_SYMBOLS.put(1e1,   "da"); // deca
        MULTIPLE_SYMBOLS.put(1.0,   "");
        MULTIPLE_SYMBOLS.put(1e-1,  "d");  // deci
        MULTIPLE_SYMBOLS.put(1e-2,  "c");  // centi
        MULTIPLE_SYMBOLS.put(1e-3,  "m");  // milli
        MULTIPLE_SYMBOLS.put(1e-6,  "µ");  // micro
        MULTIPLE_SYMBOLS.put(1e-9,  "n");  // nano
        MULTIPLE_SYMBOLS.put(1e-12, "p");  // pico
        MULTIPLE_SYMBOLS.put(1e-15, "f");  // femto
        MULTIPLE_SYMBOLS.put(1e-18, "a");  // atto
        MULTIPLE_SYMBOLS.put(1e-21, "z");  // zepto (1991)
        MULTIPLE_SYMBOLS.put(1e-24, "y");  // yocto (1991)
        MULTIPLE_SYMBOLS.put(1e-27, "r");  // ronto (2022)
        MULTIPLE_SYMBOLS.put(1e-30, "q");  // quecto (2022)
    }

    private static Component getUnitSymbol(IUnit unit) {
        if (unit instanceof Enum<?> enumUnit) {
            String unitName = unit.getClass().getSimpleName();
            return FormicApiLang.translate("units." + unitName.toLowerCase() + ".symbol." + enumUnit.name().toLowerCase()).component();
        }
        return Component.empty();
    }

    public static LangBuilder translate(String langKey, Object... args) {
        return builder().translate(langKey, args);
    }

    public static LangBuilder builder() {
        return new LangBuilder(FormicAPI.MODID);
    }

    /**
     * Formats a double value with the most appropriate SI metric prefix symbol,
     * producing a human-readable mantissa and prefix pair.
     *
     * <p>The method selects the largest prefix whose factor is less than or equal
     * to the absolute value of {@code d}, then divides {@code d} by that factor
     * to obtain the mantissa. For example:
     * <pre>
     *   numberWithSymbol(1_500_000)  →  "1.5 M"
     *   numberWithSymbol(0.000_042)  →  "42 µ"
     *   numberWithSymbol(-3_200)     →  "-3.2 k"
     * </pre>
     *
     * <p>If no floor entry exists (i.e. the value is smaller than the smallest
     * registered prefix), the smallest available prefix is used instead. If the
     * prefix table is empty, the raw formatted value is returned with a trailing
     * space and no symbol.
     *
     * <p>Prefix selection is based on the absolute value of {@code d}, so
     * negative numbers are handled symmetrically — the sign is preserved in
     * the mantissa.
     *
     * @param d the value to format; may be negative, zero, or positive
     * @return a {@link LangBuilder} containing the formatted mantissa and prefix
     *         symbol (e.g. {@code "1.5 M"}), or the raw formatted value if no
     *         suitable prefix is available
     * @see #MULTIPLE_SYMBOLS
     * @see LangNumberFormat#format(double)
     */
    public static LangBuilder numberWithSymbol(double d) {
        double abs = Math.abs(d);
        Map.Entry<Double, String> entry = MULTIPLE_SYMBOLS.floorEntry(abs);
        if (entry == null)
            entry = MULTIPLE_SYMBOLS.ceilingEntry(abs);
        if (entry == null)
            return builder().text(LangNumberFormat.format(d) + " ");
        return builder().text(LangNumberFormat.format(d / entry.getKey()) + " " + entry.getValue());
    }

    public static LangBuilder formatPressure(float pressure) {
        Pressure unit = FormicAPIConfigs.COMMON.units.pressure.get();
        return CreateLang.builder().add(Component.literal("P = "))
                .add(numberWithSymbol(unit.convert(pressure)))
                .add(getUnitSymbol(unit));
    }

    public static LangBuilder formatTemperature(float temperature) {
        Temperature unit = FormicAPIConfigs.COMMON.units.temperature.get();
        return CreateLang.builder().add(Component.literal("T = "))
                .text(LangNumberFormat.format(unit.convert(temperature)) + " ")
                .add(getUnitSymbol(unit));
    }

    public static LangBuilder formatRadiationFlux(float radiationFlux) {
        IrradiationFlux unit = FormicAPIConfigs.COMMON.units.radiationFlux.get();
        return CreateLang.builder().add(Component.literal("activity : "))
                .add(numberWithSymbol(unit.convert(radiationFlux)))
                .add(getUnitSymbol(unit));
    }
}
