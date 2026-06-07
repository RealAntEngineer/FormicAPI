package com.rae.formicapi.content.data.managers;

import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.mojang.logging.LogUtils;
import com.mojang.serialization.Codec;
import com.mojang.serialization.JsonOps;
import com.mojang.serialization.codecs.RecordCodecBuilder;
import com.rae.formicapi.fondation.math.data.SplineBased2DFunction;
import net.minecraft.client.Minecraft;
import net.minecraft.resources.ResourceLocation;
import net.minecraft.server.packs.resources.ResourceManager;
import net.minecraft.server.packs.resources.SimpleJsonResourceReloadListener;
import net.minecraft.util.GsonHelper;
import net.minecraft.util.profiling.ProfilerFiller;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;

import java.util.List;
import java.util.Map;

public class Spline2DFunctionLoader extends SimpleJsonResourceReloadListener {
    public static final  Logger                LOGGER = LogUtils.getLogger();
    //private static Map<String, TwoDSparseTabulatedFunction> FUNCTIONS_HOLDERS = ;
    public static final Codec<List<SplineBased2DFunction.Vec2>> CONTROL_POINTS_CODEC =
            SplineBased2DFunction.Vec2.CODEC.listOf();
    // Codec for a single iso-line
    public static final Codec<SplineBased2DFunction.IsoLine> ISO_LINE_CODEC = RecordCodecBuilder.create(instance ->
            instance.group(
                    Codec.FLOAT.fieldOf("value").forGetter(SplineBased2DFunction.IsoLine::getValue),
                    CONTROL_POINTS_CODEC.fieldOf("controlPoints").forGetter(SplineBased2DFunction.IsoLine::getControlPoints),
                    Codec.STRING.optionalFieldOf("splineType", "catmull_rom")
                            .forGetter(SplineBased2DFunction.IsoLine::getSplineType)
            ).apply(instance, SplineBased2DFunction.IsoLine::new)
    );
    // Codec for the entire function
    public static final Codec<SplineBased2DFunction> CODEC = RecordCodecBuilder.create(instance ->
            instance.group(
                    ISO_LINE_CODEC.listOf().fieldOf("isoLines").forGetter(SplineBased2DFunction::isoLines),
                    Codec.BOOL.optionalFieldOf("clamp", true).forGetter(SplineBased2DFunction::clamp),
                    Codec.STRING.xmap(SplineBased2DFunction.InterpolationType::valueOf, Enum::toString).optionalFieldOf("interpolationType", SplineBased2DFunction.InterpolationType.CUBIC)
                            .forGetter(SplineBased2DFunction::interpolationType)
            ).apply(instance, SplineBased2DFunction::new)
    );
    private static final Gson                  GSON   = new Gson();
    private static final String                FOLDER = "sparce_tabulated_functions";
    private final        ResourceLocation      FILE_NAME;
    private              SplineBased2DFunction FUNCTION;


    public Spline2DFunctionLoader(String modId, String fileName) {
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
                FUNCTION = CODEC.decode(JsonOps.INSTANCE, json).getOrThrow(false, s -> {
                }).getFirst();
            } catch (Exception e) {
                LOGGER.error("Failed to load float data from {}", entry.getKey(), e);
            }
        }
    }

    public float getValue(float x, float y) {
        if (loaded()) {
            return FUNCTION.evaluate(x, y);
        } else {
            boolean local = Minecraft.getInstance().isLocalServer();
            throw new RuntimeException("Function called before table could be loaded " +
                    (
                            local ?
                                    "on a local instance ??" :
                                    "on a distant machine check if you have optimisation mod preventing synchronisation"
                    ));
        }
    }

    public boolean loaded() {
        return FUNCTION != null;
    }

    /*
    public List<CompoundTag> splitSerialize(){
        List<TwoDSparseTabulatedFunction> splitTables = FUNCTION.split(1000);

        return splitTables.parallelStream().map(
                (f) -> (CompoundTag) CODEC.encode(
                                f, NbtOps.INSTANCE, new CompoundTag())
                        .getOrThrow(false, (s) -> {})
        ).toList();
    }

    public void mergeFromNBT(CompoundTag tag){
        FUNCTION = CODEC.decode(NbtOps.INSTANCE, tag)
                .getOrThrow(false, s -> {}).getFirst();
    }
    public void clearFunction(){
        FUNCTION.clear();
    }*/
}