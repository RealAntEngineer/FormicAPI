package com.rae.formicapi.math.data;

import com.mojang.serialization.Codec;
import com.mojang.serialization.codecs.RecordCodecBuilder;

import java.util.Map;
import java.util.TreeMap;
import java.util.function.BiFunction;

public class TwoDTabulatedFunction {
    // table: X -> (Y -> Value)
    private final TreeMap<Float, TreeMap<Float, Float>> table;
    private final float xStep;
    private final float yStep;
    private final StepMode xMode;
    private final StepMode yMode;
    private final boolean clamp;
    // Codec for individual inner maps (Y -> Value)
    public static final Codec<TreeMap<Float, Float>> INNER_MAP_CODEC = Codec.unboundedMap(
            Codec.STRING.xmap(Float::parseFloat,Object::toString), Codec.FLOAT
    ).xmap(TreeMap::new, TreeMap::new);

    // Codec for the entire TwoDTabulatedFunction table
    public static final Codec<TwoDTabulatedFunction> CODEC = RecordCodecBuilder.create(instance -> instance.group(
            Codec.unboundedMap(Codec.STRING.xmap(Float::parseFloat,Object::toString), INNER_MAP_CODEC).xmap(TreeMap::new, TreeMap::new).fieldOf("table")
                    .forGetter(f -> f.table),
            Codec.FLOAT.fieldOf("x_step").forGetter(f -> f.xStep),
            Codec.FLOAT.fieldOf("y_step").forGetter(f -> f.yStep),
            StepMode.CODEC.fieldOf("x_mode").forGetter(f -> f.xMode),
            StepMode.CODEC.fieldOf("y_mode").forGetter(f -> f.yMode),
            Codec.BOOL.fieldOf("clamp").forGetter(f -> f.clamp)
    ).apply(instance, TwoDTabulatedFunction::new));

    // StreamCodec example (Fabric API style)
    /*public static final StreamCodec<TwoDTabulatedFunction> STREAM_CODEC = StreamCodec.composite(
            StreamCodec.map(StreamCodec.FLOAT, StreamCodec.map(StreamCodec.FLOAT, StreamCodec.FLOAT)).fieldOf(TwoDTabulatedFunction::getTable),
            StreamCodec.FLOAT.fieldOf(f -> f.xStep),
            StreamCodec.FLOAT.fieldOf(f -> f.yStep),
            StepMode.STREAM_CODEC.fieldOf(f -> f.xMode),
            StepMode.STREAM_CODEC.fieldOf(f -> f.yMode),
            StreamCodec.BOOL.fieldOf(f -> f.clamp),
            TwoDTabulatedFunction::new
    );*/
    public TwoDTabulatedFunction(TreeMap<Float, TreeMap<Float, Float>> table, float xStep, float yStep, StepMode xMode, StepMode yMode, boolean clamp) {
        this.table = table;
        this.xStep = xStep;
        this.yStep = yStep;
        this.xMode = xMode;
        this.yMode = yMode;
        this.clamp = clamp;
    }
    public static TwoDTabulatedFunction populate(
            BiFunction<Float, Float, Float> f,
            float xStart, float yStart,
            float xEnd, float yEnd,
            int xNbr, int yNbr,
            StepMode xMode, StepMode yMode,
            boolean clamp
    ) {
        TreeMap<Float, TreeMap<Float, Float>> table = new TreeMap<>();

        float xStep = (float) ((xMode.forward.applyAsDouble(xEnd) - xMode.forward.applyAsDouble(xStart)) / (xNbr - 1));
        float yStep = (float) ((yMode.forward.applyAsDouble(yEnd) - yMode.forward.applyAsDouble(yStart)) / (yNbr - 1));

        for (float x = xStart; x <= xEnd + 1e-6; x = (float) xMode.inverse.applyAsDouble(xMode.forward.applyAsDouble(x) + xStep)) {
            TreeMap<Float, Float> row = new TreeMap<>();
            for (float y = yStart; y <= yEnd + 1e-6; y = (float) yMode.inverse.applyAsDouble(yMode.forward.applyAsDouble(y) + yStep)) {
                try {
                    float value = f.apply(x, y);
                    if (!Float.isNaN(value)) {
                        row.put(y, value);
                    }
                } catch (Exception ignored) {
                    System.out.println(ignored);
                }
            }
            table.put(x, row);
        }
        return new TwoDTabulatedFunction(table, xStep, yStep, xMode, yMode, clamp);
    }

    public float evaluate(float xInput, float yInput) {
        if (table.isEmpty()) {
            throw new IllegalStateException("Function table is empty");
        }

        float xIndex = (float) (xMode.forward.applyAsDouble(xInput) / xStep);
        int xLowerIndex = (int) Math.floor(xIndex);
        float xFrac = xIndex - xLowerIndex;

        float X1 = (float) xMode.inverse.applyAsDouble(xLowerIndex * xStep);
        float X2 = (float) xMode.inverse.applyAsDouble((xLowerIndex + 1) * xStep);

        TreeMap<Float, Float> row1 = table.get(X1);
        TreeMap<Float, Float> row2 = table.get(X2);

        if (row1 == null || row2 == null) {
            if (clamp) {
                Map.Entry<Float, TreeMap<Float, Float>> nearest = table.floorEntry(xInput);
                if (nearest == null) nearest = table.ceilingEntry(xInput);
                if (nearest == null) return table.firstEntry().getValue().firstEntry().getValue();
                return evaluate1D(yInput, nearest.getValue());
            } else {
                return extrapolateZ(xInput, yInput);
            }
        }

        float v1 = evaluate1D(yInput, row1);
        float v2 = evaluate1D(yInput, row2);

        return v1 * (1 - xFrac) + v2 * xFrac;
    }

    private float evaluate1D(float yInput, TreeMap<Float, Float> row) {
        if (row.isEmpty()) {
            throw new IllegalStateException("Row table is empty");
        }

        float yIndex = (float) (yMode.forward.applyAsDouble(yInput) / yStep);
        int yLowerIndex = (int) Math.floor(yIndex);
        float yFrac = yIndex - yLowerIndex;

        float Y1 = (float) yMode.inverse.applyAsDouble(yLowerIndex * yStep);
        float Y2 = (float) yMode.inverse.applyAsDouble((yLowerIndex + 1) * yStep);

        Float Z1 = row.get(Y1);
        Float Z2 = row.get(Y2);

        if (Z1 == null || Z2 == null) {
            Map.Entry<Float, Float> lower = row.floorEntry(yInput);
            Map.Entry<Float, Float> upper = row.ceilingEntry(yInput);

            if (lower == null || upper == null) {
                return row.firstEntry().getValue();
            }

            float T_lower = lower.getKey();
            float T_upper = upper.getKey();
            if (T_lower == T_upper) {
                return lower.getValue();
            }
            float fracAlt = (yInput - T_lower) / (T_upper - T_lower);

            return lower.getValue() * (1 - fracAlt) + upper.getValue() * fracAlt;
        }

        return Z1 * (1 - yFrac) + Z2 * yFrac;
    }

    private float extrapolateZ(float xInput, float yInput) {
        Map.Entry<Float, TreeMap<Float, Float>> lower = table.floorEntry(xInput);
        Map.Entry<Float, TreeMap<Float, Float>> upper = table.ceilingEntry(xInput);

        if (lower == null) return evaluate1D(yInput, table.firstEntry().getValue());
        if (upper == null) return evaluate1D(yInput, table.lastEntry().getValue());
        if (lower.getKey().equals(upper.getKey())) return evaluate1D(yInput, lower.getValue());

        float x1 = lower.getKey();
        float x2 = upper.getKey();
        float v1 = evaluate1D(yInput, lower.getValue());
        float v2 = evaluate1D(yInput, upper.getValue());

        float t = (xInput - x1) / (x2 - x1);
        return v1 * (1 - t) + v2 * t;
    }
}
