package com.rae.formicapi.math.data;

import java.util.Map;
import java.util.TreeMap;

public class OneDTabulatedFunction {
    private final TreeMap<Float, Float> table;
    private final float step;
    private final StepMode mode;
    private final boolean clamp;

    /**
     * @param table Sorted data: x -> y
     * @param step Distance between values in transformed space (e.g., log(x) if LOGARITHMIC)
     * @param mode The axis step mode (linear, log, etc.)
     * @param clamp Clamp outside instead of extrapolating.
     */
    public OneDTabulatedFunction(TreeMap<Float, Float> table, float step, StepMode mode, boolean clamp) {
        this.table = table;
        this.step = step;
        this.mode = mode;
        this.clamp = clamp;
    }

    public float evaluate(float input) {
        if (table.isEmpty()) {
            throw new IllegalStateException("Function table is empty");
        }

        // Handle out-of-bounds
        if (input <= table.firstKey()) {
            if (clamp) {
                return table.firstEntry().getValue();
            }
            return extrapolateBelow(input);
        } else if (input >= table.lastKey()) {
            if (clamp) {
                return table.lastEntry().getValue();
            }
            return extrapolateAbove(input);
        }

        // Interpolation inside the range
        return interpolate(input);
    }

    private float interpolate(double input) {
        float index = (float) (mode.forward.applyAsDouble(input) / step);
        int lowerIndex = (int) Math.floor(index);
        float frac = index - lowerIndex;

        float X1 = (float) mode.inverse.applyAsDouble(lowerIndex * step);
        float X2 = (float) mode.inverse.applyAsDouble((lowerIndex + 1) * step);

        // Safeguard in case floating-point precision causes a missing key
        if (!table.containsKey(X1) || !table.containsKey(X2)) {
            Map.Entry<Float, Float> lower = table.floorEntry((float) input);
            Map.Entry<Float, Float> upper = table.ceilingEntry((float) input);

            if (lower == null || upper == null) {
                return table.get(table.firstKey()); // fallback
            }

            float T_lower = lower.getKey();
            float T_upper = upper.getKey();
            if (T_lower == T_upper) {
                return lower.getValue();
            }
            float fracAlt = (float) ((input - T_lower) / (T_upper - T_lower));
            return lower.getValue() * (1 - fracAlt) + upper.getValue() * fracAlt;
        }

        float P1 = table.get(X1);
        float P2 = table.get(X2);

        return P1 * (1 - frac) + P2 * frac;
    }

    private float extrapolateBelow(float query) {
        Map.Entry<Float, Float> lower = table.firstEntry();
        Map.Entry<Float, Float> upper = table.higherEntry(lower.getKey());
        if (upper == null) return lower.getValue(); // only one point in table
        return linear(query, lower, upper);
    }

    private float extrapolateAbove(float query) {
        Map.Entry<Float, Float> upper = table.lastEntry();
        Map.Entry<Float, Float> lower = table.lowerEntry(upper.getKey());
        if (lower == null) return upper.getValue(); // only one point in table
        return linear(query, lower, upper);
    }

    private float linear(float query, Map.Entry<Float, Float> a, Map.Entry<Float, Float> b) {
        float x1 = a.getKey();
        float x2 = b.getKey();
        float y1 = a.getValue();
        float y2 = b.getValue();
        float t = (query - x1) / (x2 - x1);
        return y1 * (1 - t) + y2 * t;
    }
}

