package com.rae.formicapi.data.managers;

import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.mojang.logging.LogUtils;
import com.mojang.serialization.JsonOps;
import com.rae.formicapi.math.data.TwoDSparseTabulatedFunction;
import net.minecraft.resources.ResourceLocation;
import net.minecraft.server.packs.resources.ResourceManager;
import net.minecraft.server.packs.resources.SimpleJsonResourceReloadListener;
import net.minecraft.util.GsonHelper;
import net.minecraft.util.profiling.ProfilerFiller;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;

import java.util.Map;

public class TwoDSparceTabulatedFunctionLoader extends SimpleJsonResourceReloadListener {
    private static final Gson GSON = new Gson();
    private static final String FOLDER = "sparce_tabulated_functions";

    private final ResourceLocation FILE_NAME;
    private TwoDSparseTabulatedFunction FUNCTION;
    public static final Logger LOGGER = LogUtils.getLogger();
    public TwoDSparceTabulatedFunctionLoader(String modId, String fileName) {
        super(GSON, FOLDER);
        FILE_NAME = new ResourceLocation(modId, fileName);
    }

    @Override
    protected void apply(Map<ResourceLocation, JsonElement> map, @NotNull ResourceManager resourceManager, @NotNull ProfilerFiller profiler) {
        LOGGER.info("Reloading TwoDTabulatedFunctionLoader for: {}", FILE_NAME);

        for (Map.Entry<ResourceLocation, JsonElement> entry : map.entrySet()) {
            if (!entry.getKey().equals(FILE_NAME)) continue;
            try {
                JsonObject json = GsonHelper.convertToJsonObject(entry.getValue(), "sparce tabulated function");
                FUNCTION = TwoDSparseTabulatedFunction.CODEC.decode(JsonOps.INSTANCE, json).getOrThrow(false, s -> {}).getFirst();
            } catch (Exception e) {
                LOGGER.error("Failed to load float data from {}", entry.getKey(), e);
            }
        }
    }

    public float getValue(float x,float y) {
        return FUNCTION.evaluate(x, y);
    }

    public boolean loaded() {
        return FUNCTION != null;
    }


}