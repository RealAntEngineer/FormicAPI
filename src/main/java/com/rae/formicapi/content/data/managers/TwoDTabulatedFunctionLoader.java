package com.rae.formicapi.content.data.managers;

import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.mojang.logging.LogUtils;
import com.mojang.serialization.Codec;
import com.mojang.serialization.JsonOps;
import com.mojang.serialization.codecs.RecordCodecBuilder;
import com.rae.formicapi.fondation.math.data.StepMode;
import com.rae.formicapi.fondation.math.data.TwoDTabulatedFunction;
import net.minecraft.resources.ResourceLocation;
import net.minecraft.server.packs.resources.ResourceManager;
import net.minecraft.server.packs.resources.SimpleJsonResourceReloadListener;
import net.minecraft.util.GsonHelper;
import net.minecraft.util.profiling.ProfilerFiller;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;

import java.util.Map;
import java.util.TreeMap;

public class TwoDTabulatedFunctionLoader extends SimpleJsonResourceReloadListener {
    public static final Logger LOGGER = LogUtils.getLogger();
    private static final Gson GSON = new Gson();
    private static final String FOLDER = "tabulated_functions";
    private final ResourceLocation FILE_NAME;
    private TwoDTabulatedFunction FUNCTION;

    // Codec for individual inner maps (Y -> Value)
    public static final Codec<TreeMap<Float, Float>> INNER_MAP_CODEC = Codec.unboundedMap(
            Codec.STRING.xmap(Float::parseFloat, Object::toString), Codec.FLOAT
    ).xmap(TreeMap::new, TreeMap::new);
    // Codec for the entire TwoDTabulatedFunction table
    public static final Codec<TwoDTabulatedFunction> CODEC           = RecordCodecBuilder.create(instance -> instance.group(
            Codec.unboundedMap(Codec.STRING.xmap(Float::parseFloat, Object::toString), INNER_MAP_CODEC).xmap(TreeMap::new, TreeMap::new).fieldOf("table")
                    .forGetter(TwoDTabulatedFunction::table),
            Codec.FLOAT.fieldOf("x_step").forGetter(TwoDTabulatedFunction::xStep),
            Codec.FLOAT.fieldOf("y_step").forGetter(TwoDTabulatedFunction::yStep),
            StepMode.CODEC.fieldOf("x_mode").forGetter(TwoDTabulatedFunction::xMode),
            StepMode.CODEC.fieldOf("y_mode").forGetter(TwoDTabulatedFunction::yMode),
            Codec.BOOL.fieldOf("clamp").forGetter(TwoDTabulatedFunction::clamp)
    ).apply(instance, TwoDTabulatedFunction::new));

    public TwoDTabulatedFunctionLoader(String modId, String fileName) {
        super(GSON, FOLDER);
        FILE_NAME = new ResourceLocation(modId, fileName);
    }

    @Override
    protected void apply(Map<ResourceLocation, JsonElement> map, @NotNull ResourceManager resourceManager, @NotNull ProfilerFiller profiler) {
        LOGGER.info("Reloading TwoDTabulatedFunctionLoader for: {}", FILE_NAME);

        for (Map.Entry<ResourceLocation, JsonElement> entry : map.entrySet()) {
            if (!entry.getKey().equals(FILE_NAME)) continue;
            try {
                JsonObject json = GsonHelper.convertToJsonObject(entry.getValue(), "tabulated function");
                FUNCTION = CODEC.decode(JsonOps.INSTANCE, json).getOrThrow(false, s -> {
                }).getFirst();
            } catch (Exception e) {
                LOGGER.error("Failed to load float data from {}", entry.getKey(), e);
            }
        }
    }

    public float getValue(float x, float y) {
        return FUNCTION.evaluate(x, y);
    }

    public boolean loaded() {
        return FUNCTION != null;
    }


}