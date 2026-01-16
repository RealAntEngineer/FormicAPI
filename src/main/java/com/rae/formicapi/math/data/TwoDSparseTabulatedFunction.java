package com.rae.formicapi.math.data;

import com.mojang.serialization.Codec;
import com.mojang.serialization.codecs.RecordCodecBuilder;

import java.util.Map;
import java.util.TreeMap;
import java.util.function.BiFunction;

public class TwoDSparseTabulatedFunction {
    // table: X -> (Y -> Value)
    private final TreeMap<Float, TreeMap<Float, Float>> table;

    private final boolean clamp;
    // Codec for individual inner maps (Y -> Value)
    public static final Codec<TreeMap<Float, Float>> INNER_MAP_CODEC = Codec.unboundedMap(
            Codec.STRING.xmap(Float::parseFloat,Object::toString), Codec.FLOAT
    ).xmap(TreeMap::new, TreeMap::new);

    // Codec for the entire TwoDTabulatedFunction table
    public static final Codec<TwoDSparseTabulatedFunction> CODEC = RecordCodecBuilder.create(instance -> instance.group(
            Codec.unboundedMap(Codec.STRING.xmap(Float::parseFloat,Object::toString), INNER_MAP_CODEC).xmap(TreeMap::new, TreeMap::new).fieldOf("table")
                    .forGetter(f -> f.table),
            Codec.BOOL.fieldOf("clamp").forGetter(f -> f.clamp)
    ).apply(instance, TwoDSparseTabulatedFunction::new));

    public TwoDSparseTabulatedFunction(TreeMap<Float, TreeMap<Float, Float>> table, boolean clamp) {
        this.table = table;
        this.clamp = clamp;
    }

    public float evaluate(float xInput, float yInput) {
        if (table.isEmpty()) {
            throw new IllegalStateException("Function table is empty");
        }

        // Get nearest X bounds
        Map.Entry<Float, TreeMap<Float, Float>> lowerX = table.floorEntry(xInput);
        Map.Entry<Float, TreeMap<Float, Float>> upperX = table.ceilingEntry(xInput);

        if (lowerX == null && upperX == null) {
            throw new IllegalStateException("No data in table at all");
        }
        if (lowerX == null) {
            return clamp ? evaluate1D(yInput, upperX.getValue())
                    : extrapolateZ(xInput, yInput);
        }
        if (upperX == null) {
            return clamp ? evaluate1D(yInput, lowerX.getValue())
                    : extrapolateZ(xInput, yInput);
        }
        if (lowerX.getKey().equals(upperX.getKey())) {
            return evaluate1D(yInput, lowerX.getValue());
        }

        // Interpolate across X
        float x1 = lowerX.getKey();
        float x2 = upperX.getKey();
        float v1 = evaluate1D(yInput, lowerX.getValue());
        float v2 = evaluate1D(yInput, upperX.getValue());
        float t = (xInput - x1) / (x2 - x1);
        return v1 * (1 - t) + v2 * t;
    }

    private float evaluate1D(float yInput, TreeMap<Float, Float> row) {
        if (row.isEmpty()) {
            throw new IllegalStateException("Row table is empty");
        }

        // Find nearest Y bounds
        Map.Entry<Float, Float> lowerY = row.floorEntry(yInput);
        Map.Entry<Float, Float> upperY = row.ceilingEntry(yInput);

        if (lowerY == null && upperY == null) {
            throw new IllegalStateException("No data in row at all");
        }
        if (lowerY == null) {
            return clamp ? upperY.getValue() : extrapolateY(yInput, row);
        }
        if (upperY == null) {
            return clamp ? lowerY.getValue() : extrapolateY(yInput, row);
        }
        if (lowerY.getKey().equals(upperY.getKey())) {
            return lowerY.getValue();
        }

        // Interpolate across Y
        float y1 = lowerY.getKey();
        float y2 = upperY.getKey();
        float v1 = lowerY.getValue();
        float v2 = upperY.getValue();
        float t = (yInput - y1) / (y2 - y1);
        return v1 * (1 - t) + v2 * t;
    }

    private float extrapolateY(float yInput, TreeMap<Float, Float> row) {
        Map.Entry<Float, Float> lower = row.floorEntry(yInput);
        Map.Entry<Float, Float> upper = row.ceilingEntry(yInput);

        if (lower == null) {
            // extrapolate below using first two points
            Map.Entry<Float, Float> first = row.firstEntry();
            Map.Entry<Float, Float> next = row.higherEntry(first.getKey());
            if (next == null) return first.getValue();
            return linear(yInput, first, next);
        }
        if (upper == null) {
            // extrapolate above using last two points
            Map.Entry<Float, Float> last = row.lastEntry();
            Map.Entry<Float, Float> prev = row.lowerEntry(last.getKey());
            if (prev == null) return last.getValue();
            return linear(yInput, prev, last);
        }
        // Already handled in evaluate1D, should not reach here
        return lower.getValue();
    }

    private float linear(float query, Map.Entry<Float, Float> a, Map.Entry<Float, Float> b) {
        float x1 = a.getKey();
        float x2 = b.getKey();
        float y1 = a.getValue();
        float y2 = b.getValue();
        float t = (query - x1) / (x2 - x1);
        return y1 * (1 - t) + y2 * t;
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
