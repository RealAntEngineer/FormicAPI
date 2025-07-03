package com.rae.formicapi.data.providers;

import com.google.gson.JsonElement;
import com.mojang.serialization.JsonOps;
import com.rae.formicapi.math.data.StepMode;
import com.rae.formicapi.math.data.TwoDTabulatedFunction;
import net.minecraft.data.CachedOutput;
import net.minecraft.data.DataProvider;
import net.minecraft.data.PackOutput;
import net.minecraft.resources.ResourceLocation;
import org.jetbrains.annotations.NotNull;

import java.nio.file.Path;
import java.util.concurrent.CompletableFuture;
import java.util.function.BiFunction;

public class TwoDTabulatedFunctionProvider implements DataProvider {

    private final PackOutput output;
    private final ResourceLocation location;
    private final BiFunction<Float, Float, Float> f;
    private final float xStart;
    private final float yStart;
    private final float xEnd;
    private final float yEnd;
    private final int xNbr;
    private final int yNbr;
    private final StepMode xMode;
    private final StepMode yMode;
    private final boolean clamp;

    public TwoDTabulatedFunctionProvider(PackOutput output, ResourceLocation location, BiFunction<Float, Float, Float> f,
                                         float xStart, float yStart,
                                         float xEnd, float yEnd,
                                         int xNbr, int yNbr,
                                         StepMode xMode, StepMode yMode,
                                         boolean clamp
    ) {
        this.output = output;
        this.location = location;
        this.f = f;
        this.xStart = xStart;
        this.yStart = yStart;
        this.xEnd = xEnd;
        this.yEnd = yEnd;
        this.xNbr = xNbr;
        this.yNbr = yNbr;
        this.xMode = xMode;
        this.yMode = yMode;
        this.clamp = clamp;
    }

    @Override
    public @NotNull CompletableFuture<?> run(@NotNull CachedOutput cache) {
        // Example path: data/yourmod/tabulated_functions/my_function.json
        TwoDTabulatedFunction function = TwoDTabulatedFunction.populate(
              f, // your function here
                xStart, yStart, xEnd, yEnd, xNbr, yNbr,
                xMode, yMode,
                clamp
        );

        JsonElement json = TwoDTabulatedFunction.CODEC.encodeStart(JsonOps.INSTANCE, function)
                .getOrThrow(false, s -> {
                });
        Path path = output.getOutputFolder()
                .resolve("data/" + location.getNamespace() + "/tabulated_functions/" + location.getPath() + ".json");

        return DataProvider.saveStable(cache, json, path);
    }

    @Override
    public @NotNull String getName() {
        return "2D Functions for " + location.toString();
    }
}