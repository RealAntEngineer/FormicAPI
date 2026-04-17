package com.rae.formicapi.content.data.managers;

import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.mojang.logging.LogUtils;
import com.mojang.serialization.JsonOps;
import com.rae.formicapi.fondation.math.data.TwoDSparseTabulatedFunction;
import net.minecraft.client.Minecraft;
import net.minecraft.nbt.CompoundTag;
import net.minecraft.nbt.NbtOps;
import net.minecraft.resources.ResourceLocation;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.packs.resources.ResourceManager;
import net.minecraft.server.packs.resources.SimpleJsonResourceReloadListener;
import net.minecraft.util.GsonHelper;
import net.minecraft.util.profiling.ProfilerFiller;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;

import java.util.Map;

public class TwoDSparceTabulatedFunctionLoader extends SimpleJsonResourceReloadListener {
    public static final Logger LOGGER = LogUtils.getLogger();
    private static final Gson GSON = new Gson();
    private static final String FOLDER = "sparce_tabulated_functions";
    private final ResourceLocation FILE_NAME;
    private TwoDSparseTabulatedFunction FUNCTION;

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
                FUNCTION = TwoDSparseTabulatedFunction.CODEC.decode(JsonOps.INSTANCE, json).getOrThrow(false, s -> {
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
        return FUNCTION != null;
    }

    public CompoundTag serialize(){
        return (CompoundTag) TwoDSparseTabulatedFunction.CODEC.encode(
                FUNCTION, NbtOps.INSTANCE, new CompoundTag())
                .getOrThrow(false, (s) -> {});
    }

    public void reloadFromNBT(CompoundTag tag){
        FUNCTION = TwoDSparseTabulatedFunction.CODEC.decode(NbtOps.INSTANCE, tag)
                .getOrThrow(false, s -> {}).getFirst();
    }
}