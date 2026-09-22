package com.rae.formicapi.content.data.managers;

import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.mojang.logging.LogUtils;
import com.mojang.serialization.Codec;
import com.mojang.serialization.JsonOps;
import com.mojang.serialization.codecs.RecordCodecBuilder;
import com.rae.formicapi.FormicAPI;
import com.rae.formicapi.foundation.math.data.TwoDSparseTabulatedFunction;
import net.minecraft.client.Minecraft;
import net.minecraft.nbt.CompoundTag;
import net.minecraft.nbt.NbtOps;
import net.minecraft.resources.ResourceLocation;
import net.minecraft.server.packs.resources.ResourceManager;
import net.minecraft.server.packs.resources.SimpleJsonResourceReloadListener;
import net.minecraft.util.GsonHelper;
import net.minecraft.util.profiling.ProfilerFiller;
import org.jetbrains.annotations.Nullable;
import org.slf4j.Logger;

import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import static com.rae.formicapi.content.data.CodecUtil.orderedMapCodec;

public class TwoDSparceTabulatedFunctionLoader extends SimpleJsonResourceReloadListener {
    public static final Logger                             LOGGER = LogUtils.getLogger();
    public static final Codec<TwoDSparseTabulatedFunction> CODEC  = RecordCodecBuilder.create(instance -> instance.group(
            orderedMapCodec(Codec.FLOAT, orderedMapCodec(Codec.FLOAT, Codec.FLOAT))
                    .xmap(TreeMap::new, TreeMap::new).fieldOf("table")
                    .forGetter(TwoDSparseTabulatedFunction::getTableCopy),
            Codec.BOOL.fieldOf("clamp").forGetter(TwoDSparseTabulatedFunction::isClamp)
    ).apply(instance, TwoDSparseTabulatedFunction::new));

    private static final Gson                        GSON     = new Gson();
    private static final String                      FOLDER   = "sparce_tabulated_functions";
    private final        ResourceLocation            FILE_NAME;
    private @Nullable    TwoDSparseTabulatedFunction FUNCTION = null;

    public TwoDSparceTabulatedFunctionLoader(String modId, String fileName) {
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
            return FUNCTION.evaluate(x, y);
        } else {
            boolean local = Minecraft.getInstance().isLocalServer();
            throw new RuntimeException("Function called before table could be loaded " +
                    (
                            local ?
                                    "on a local instance ?? Major bug, please report to mod owner" :
                                    "on a distant machine ; check if you have optimisation mod preventing synchronisation"
                    ));
        }
    }

    public boolean loaded() {
        return FUNCTION != null && !FUNCTION.isEmpty();
    }

    public List<CompoundTag> splitSerialize() {
        if (!loaded()) throw new IllegalStateException("Can't call splitSerialize if Function not loaded yet");
        assert FUNCTION != null;
        List<TwoDSparseTabulatedFunction> splitTables = FUNCTION.split(1000);

        return splitTables.parallelStream().map(
                (f) -> (CompoundTag) CODEC.encode(
                                f, NbtOps.INSTANCE, new CompoundTag())
                        .getOrThrow()
        ).toList();
    }

    public void mergeFromNBT(CompoundTag tag) {
        if (FUNCTION == null)
            FUNCTION = CODEC.decode(NbtOps.INSTANCE, tag)
                    .getOrThrow().getFirst();
        else {
            FUNCTION.mergeFrom(CODEC.decode(NbtOps.INSTANCE, tag)
                    .getOrThrow().getFirst(), false);
        }
    }

    public void clearFunction() {
        if (FUNCTION != null) {
            FUNCTION.clear();
        } else {
            FormicAPI.LOGGER.debug("Useless call to clear a table : {} the table hasn't being loaded yet, " +
                    "the player is probably joining a world for the first time in this session", FILE_NAME);
        }
    }
}