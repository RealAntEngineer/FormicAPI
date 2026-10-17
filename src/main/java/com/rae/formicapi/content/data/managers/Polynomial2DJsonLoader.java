package com.rae.formicapi.content.data.managers;

import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.mojang.logging.LogUtils;
import com.mojang.serialization.Codec;
import com.mojang.serialization.JsonOps;
import com.mojang.serialization.codecs.RecordCodecBuilder;
import com.rae.formicapi.content.thermal_utilities.Polynomial2D;
import net.minecraft.client.Minecraft;
import net.minecraft.resources.ResourceLocation;
import net.minecraft.server.packs.resources.ResourceManager;
import net.minecraft.server.packs.resources.SimpleJsonResourceReloadListener;
import net.minecraft.util.GsonHelper;
import net.minecraft.util.profiling.ProfilerFiller;
import org.jetbrains.annotations.Nullable;
import org.slf4j.Logger;

import java.util.Arrays;
import java.util.Map;

public class Polynomial2DJsonLoader extends SimpleJsonResourceReloadListener {
    public static final Logger                             LOGGER = LogUtils.getLogger();
    Codec<Polynomial2D> CODEC = RecordCodecBuilder.create(instance -> instance.group(
            Codec.INT.fieldOf("order").forGetter(Polynomial2D::getOrder),
            Codec.list(Codec.DOUBLE).xmap(
                    list -> list.stream().mapToDouble(Double::doubleValue).toArray(),
                    array -> Arrays.stream(array).boxed().toList()
            ).fieldOf("coefficients").forGetter(Polynomial2D::getCoefficients),
            Codec.DOUBLE.fieldOf("xCenter").forGetter(Polynomial2D::getxCenter),
            Codec.DOUBLE.fieldOf("xScale").forGetter(Polynomial2D::getxScale),
            Codec.DOUBLE.fieldOf("yCenter").forGetter(Polynomial2D::getyCenter),
            Codec.DOUBLE.fieldOf("yScale").forGetter(Polynomial2D::getyScale)
    ).apply(instance, Polynomial2D::new));

    private static final Gson                        GSON     = new Gson();
    private static final String                      FOLDER   = "polynomial_2d";
    private final        ResourceLocation            FILE_NAME;
    private @Nullable    Polynomial2D FUNCTION = null;

    public Polynomial2DJsonLoader(String modId, String fileName) {
        super(GSON, FOLDER);
        FILE_NAME = ResourceLocation.fromNamespaceAndPath(modId, fileName);
    }

    @Override
    protected void apply(Map<ResourceLocation, JsonElement> map, ResourceManager resourceManager, ProfilerFiller profiler) {
        LOGGER.info("Reloading TwoDTabulatedFunctionLoader for: {}", FILE_NAME);

        for (Map.Entry<ResourceLocation, JsonElement> entry : map.entrySet()) {
            if (!entry.getKey().equals(FILE_NAME)) continue;
            try {
                JsonObject json = GsonHelper.convertToJsonObject(entry.getValue(), "sparce tabulated function");
                FUNCTION = CODEC.decode(JsonOps.INSTANCE, json).getOrThrow().getFirst();
            } catch (Exception e) {
                LOGGER.error("Failed to load float data from {}", entry.getKey(), e);
            }
        }
    }

    public float getValue(float x, float y) {
        if (loaded()) {
            assert FUNCTION != null;
            return (float) FUNCTION.evaluate(x, y);
        } else {
            boolean local = Minecraft.getInstance().isLocalServer();
            throw new RuntimeException("Function called before polynome could be loaded " +
                    (
                            local ?
                                    "on a local instance ?? Major bug, please report to mod owner" :
                                    "on a distant machine ; check if you have optimisation mod preventing synchronisation"
                    ));
        }
    }

    public boolean loaded() {
        return FUNCTION != null;
    }
}