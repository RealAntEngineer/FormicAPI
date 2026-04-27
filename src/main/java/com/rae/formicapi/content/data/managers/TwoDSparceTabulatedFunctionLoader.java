package com.rae.formicapi.content.data.managers;

import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.mojang.logging.LogUtils;
import com.mojang.serialization.JsonOps;
import com.rae.formicapi.FormicAPI;
import com.rae.formicapi.fondation.math.data.TwoDSparseTabulatedFunction;
import net.minecraft.client.Minecraft;
import net.minecraft.nbt.CompoundTag;
import net.minecraft.nbt.NbtOps;
import net.minecraft.resources.ResourceLocation;
import net.minecraft.server.packs.resources.ResourceManager;
import net.minecraft.server.packs.resources.SimpleJsonResourceReloadListener;
import net.minecraft.util.GsonHelper;
import net.minecraft.util.profiling.ProfilerFiller;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;

import java.util.List;
import java.util.Map;

public class TwoDSparceTabulatedFunctionLoader extends SimpleJsonResourceReloadListener {
    public static final Logger                      LOGGER = LogUtils.getLogger();
    private static final Gson   GSON   = new Gson();
    private static final String FOLDER = "sparce_tabulated_functions";
    private final       ResourceLocation            FILE_NAME;
    private             TwoDSparseTabulatedFunction FUNCTION;

    public TwoDSparceTabulatedFunctionLoader(String modId, String fileName) {
        super(GSON, FOLDER);
        FILE_NAME = ResourceLocation.fromNamespaceAndPath(modId, fileName);
    }

    @Override
    protected void apply(Map<ResourceLocation, JsonElement> map, @NotNull ResourceManager resourceManager, @NotNull ProfilerFiller profiler) {
        LOGGER.info("Reloading TwoDTabulatedFunctionLoader for: {}", FILE_NAME);

        for (Map.Entry<ResourceLocation, JsonElement> entry : map.entrySet()) {
            if (!entry.getKey().equals(FILE_NAME)) continue;
            try {
                JsonObject json = GsonHelper.convertToJsonObject(entry.getValue(), "sparce tabulated function");
                FUNCTION = TwoDSparseTabulatedFunction.CODEC.decode(JsonOps.INSTANCE, json).getOrThrow().getFirst();
            } catch (Exception e) {
                LOGGER.error("Failed to load float data from {}", entry.getKey(), e);
            }
        }
    }

    public float getValue(float x, float y) {
        if (loaded()) {
            return FUNCTION.evaluate(x, y);
        } else {
            boolean local  = Minecraft.getInstance().isLocalServer();
            throw new RuntimeException("Function called before table could be loaded " +
                    (
                            local ?
                                    "on a local instance ??":
                                    "on a distant machine check if you have optimisation mod preventing synchronisation"
                    ));
        }
    }

    public boolean loaded() {
        return FUNCTION != null && !FUNCTION.isEmpty();
    }

    public List<CompoundTag> splitSerialize(){
        List<TwoDSparseTabulatedFunction> splitTables = FUNCTION.split(1000);

        return splitTables.parallelStream().map(
                (f) -> (CompoundTag) TwoDSparseTabulatedFunction.CODEC.encode(
                                f, NbtOps.INSTANCE, new CompoundTag())
                        .getOrThrow()
        ).toList();
    }

    public void mergeFromNBT(CompoundTag tag){
        if (FUNCTION == null)
            FUNCTION = TwoDSparseTabulatedFunction.CODEC.decode(NbtOps.INSTANCE, tag)
                .getOrThrow().getFirst();
        else {
            FUNCTION.mergeFrom(TwoDSparseTabulatedFunction.CODEC.decode(NbtOps.INSTANCE, tag)
                    .getOrThrow().getFirst(), false);
        }
    }
    public void clearFunction(){
        if (FUNCTION!=null) {
            FUNCTION.clear();
        }
        else {
            FormicAPI.LOGGER.debug("Useless call to clear a table : {} the table hasn't being loaded yet, " +
                    "the player is probably joining a world for the first time in this session", FILE_NAME);
        }
    }
}