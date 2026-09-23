package com.rae.formicapi.foundation.math.data;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.function.BiFunction;

public class TwoDTabulatedFunction {
    // Codec for individual inner maps (Y -> Value)
    //TODO go to full grid of double/float array why use step otherwise ?
    // table: X -> (Y -> Value)
    private final TreeMap<Float, TreeMap<Float, Float>> table;
    private final float                                 xStep;
    private final float                                 yStep;
    private final StepMode                              xMode;
    private final StepMode                              yMode;
    private final boolean                               clamp;

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

    public TreeMap<Float, TreeMap<Float, Float>> getTableCopy() {
        return new TreeMap<>(table);
    }

    public float getxStep() {
        return xStep;
    }

    public float getyStep() {
        return yStep;
    }

    public StepMode getxMode() {
        return xMode;
    }

    public StepMode getyMode() {
        return yMode;
    }

    public boolean isClamp() {
        return clamp;
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
        float t  = (xInput - x1) / (x2 - x1);
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
        float t  = (yInput - y1) / (y2 - y1);
        return v1 * (1 - t) + v2 * t;
    }

    private float extrapolateY(float yInput, TreeMap<Float, Float> row) {
        Map.Entry<Float, Float> lower = row.floorEntry(yInput);
        Map.Entry<Float, Float> upper = row.ceilingEntry(yInput);

        if (lower == null) {
            // extrapolate below using first two points
            Map.Entry<Float, Float> first = row.firstEntry();
            Map.Entry<Float, Float> next  = row.higherEntry(first.getKey());
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
        float t  = (query - x1) / (x2 - x1);
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

    public void mergeFrom(TwoDTabulatedFunction other, boolean overwrite) {
        for (Map.Entry<Float, TreeMap<Float, Float>> xEntry : other.table.entrySet()) {
            float                 x        = xEntry.getKey();
            TreeMap<Float, Float> otherRow = xEntry.getValue();

            TreeMap<Float, Float> thisRow = this.table.computeIfAbsent(x, k -> new TreeMap<>());

            for (Map.Entry<Float, Float> yEntry : otherRow.entrySet()) {
                if (overwrite || !thisRow.containsKey(yEntry.getKey())) {
                    thisRow.put(yEntry.getKey(), yEntry.getValue());
                }
            }
        }
    }

    public List<TwoDTabulatedFunction> split(int maxElements) {
        List<TwoDTabulatedFunction> result = new ArrayList<>();

        TreeMap<Float, TreeMap<Float, Float>> current = new TreeMap<>();
        int                                   count   = 0;

        for (Map.Entry<Float, TreeMap<Float, Float>> entry : table.entrySet()) {
            current.put(entry.getKey(), entry.getValue());
            count += entry.getValue().size();

            if (count >= maxElements) {
                result.add(new TwoDTabulatedFunction(new TreeMap<>(current), xStep, yStep, xMode, yMode, clamp));
                current.clear();
                count = 0;
            }
        }

        if (!current.isEmpty()) {
            result.add(new TwoDTabulatedFunction(new TreeMap<>(current), xStep, yStep, xMode, yMode,clamp));
        }

        return result;
    }

    public void clear() {
        table.clear();
    }

    public boolean isEmpty() {
        return table.isEmpty();
    }
}
