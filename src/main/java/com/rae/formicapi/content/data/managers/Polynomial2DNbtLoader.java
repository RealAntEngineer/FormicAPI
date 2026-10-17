package com.rae.formicapi.content.data.managers;

import com.mojang.logging.LogUtils;
import com.mojang.serialization.Codec;
import com.mojang.serialization.codecs.RecordCodecBuilder;
import com.rae.formicapi.content.thermal_utilities.Polynomial2D;
import net.minecraft.client.Minecraft;
import net.minecraft.nbt.CompoundTag;
import net.minecraft.nbt.NbtAccounter;
import net.minecraft.nbt.NbtIo;
import net.minecraft.nbt.NbtOps;
import net.minecraft.resources.ResourceLocation;
import net.minecraft.server.packs.resources.Resource;
import net.minecraft.server.packs.resources.ResourceManager;
import net.minecraft.server.packs.resources.SimplePreparableReloadListener;
import net.minecraft.util.profiling.ProfilerFiller;
import org.jetbrains.annotations.Nullable;
import org.slf4j.Logger;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class Polynomial2DNbtLoader extends SimplePreparableReloadListener<Map<ResourceLocation, CompoundTag>> {
    public static final Logger LOGGER = LogUtils.getLogger();

    private static final String FOLDER = "polynomial_2d";

    private static final Codec<Polynomial2D> CODEC = RecordCodecBuilder.create(instance -> instance.group(
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

    private final ResourceLocation FILE_NAME;
    private @Nullable Polynomial2D FUNCTION;

    public Polynomial2DNbtLoader(String modId, String fileName) {
        FILE_NAME = ResourceLocation.fromNamespaceAndPath(modId, fileName);
    }

    @Override
    protected Map<ResourceLocation, CompoundTag> prepare(ResourceManager resourceManager, ProfilerFiller profiler) {
        Map<ResourceLocation, CompoundTag> result = new HashMap<>();

        for (Map.Entry<ResourceLocation, Resource> entry :
                resourceManager.listResources(FOLDER, path -> path.getPath().endsWith(".nbt")).entrySet()) {
            try {
                CompoundTag tag = NbtIo.readCompressed(
                        entry.getValue().open(),
                        NbtAccounter.unlimitedHeap()
                );

                result.put(entry.getKey(), tag);
            } catch (Exception e) {
                LOGGER.error("Failed to load NBT data from {}", entry.getKey(), e);
            }
        }

        return result;
    }

    @Override
    protected void apply(Map<ResourceLocation, CompoundTag> map, ResourceManager resourceManager, ProfilerFiller profiler) {
        LOGGER.info("Reloading Polynomial2DNbtLoader for: {}", FILE_NAME);

        CompoundTag tag = map.get(FILE_NAME);
        if (tag == null) {
            LOGGER.error("Could not find {}", FILE_NAME);
            FUNCTION = null;
            return;
        }

        try {
            FUNCTION = CODEC.decode(NbtOps.INSTANCE, tag).getOrThrow().getFirst();
        } catch (Exception e) {
            LOGGER.error("Failed to decode polynomial from {}", FILE_NAME, e);
            FUNCTION = null;
        }
    }

    public float getValue(float x, float y) {
        if (loaded()) {
            assert FUNCTION != null;
            return (float) FUNCTION.evaluate(x, y);
        }

        boolean local = Minecraft.getInstance().isLocalServer();
        throw new RuntimeException("Function called before polynomial could be loaded " +
                (local
                        ? "on a local instance ?? Major bug, please report to mod owner"
                        : "on a distant machine ; check if you have optimisation mod preventing synchronisation"));
    }

    public boolean loaded() {
        return FUNCTION != null;
    }
}