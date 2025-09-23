package com.rae.formicapi;

import com.rae.formicapi.math.data.StepMode;
import com.rae.formicapi.math.data.TwoDTabulatedFunction;
import org.junit.jupiter.api.Test;

import java.util.Random;
import java.util.TreeMap;

import static org.junit.jupiter.api.Assertions.*;

class TwoDTabulatedFunctionTest {

    private TwoDTabulatedFunction makeFunc(boolean clamp) {
        // z(x,y) = x + y, simple additive function
        TreeMap<Float, TreeMap<Float, Float>> table = new TreeMap<>();
        for (int xi = 0; xi <= 5; xi++) {
            TreeMap<Float, Float> row = new TreeMap<>();
            for (int yi = 0; yi <= 5; yi++) {
                row.put((float) yi, (float) (xi + yi));
            }
            table.put((float) xi, row);
        }
        return new TwoDTabulatedFunction(table, 1f, 1f, StepMode.LINEAR, StepMode.LINEAR, clamp);
    }

    private TwoDTabulatedFunction makeFuncWithHoles(boolean clamp) {
        // z(x,y) = x + y, but missing some rows and entries
        TreeMap<Float, TreeMap<Float, Float>> table = new TreeMap<>();
        for (int xi = 0; xi <= 5; xi++) {
            if (xi == 2) continue; // skip entire row (hole in X)
            TreeMap<Float, Float> row = new TreeMap<>();
            for (int yi = 0; yi <= 5; yi++) {
                if (yi == 3) continue; // skip inner point (hole in Y)
                row.put((float) yi, (float) (xi + yi));
            }
            table.put((float) xi, row);
        }
        return new TwoDTabulatedFunction(table, 1f, 1f, StepMode.LINEAR, StepMode.LINEAR, clamp);
    }

    @Test
    void testExactPoints() {
        TwoDTabulatedFunction f = makeFunc(false);
        for (int x = 0; x <= 5; x++) {
            for (int y = 0; y <= 5; y++) {
                assertEquals(x + y, f.evaluate(x, y), 1e-6,
                        "Exact point failed at (" + x + "," + y + ")");
            }
        }
    }

    @Test
    void testMidpoints() {
        TwoDTabulatedFunction f = makeFunc(false);
        float val = f.evaluate(2.5f, 3.5f);
        assertEquals(2.5f + 3.5f, val, 1e-6, "Midpoint interpolation failed");
    }

    @Test
    void testExtrapolationBelowAndAbove() {
        TwoDTabulatedFunction f = makeFunc(false);
        assertEquals(-1f, f.evaluate(-1f, 0f), 1e-6, "Extrapolation below X failed");
        assertEquals(11f, f.evaluate(5f, 6f), 1e-6, "Extrapolation above Y failed");
        assertEquals(12f, f.evaluate(6f, 6f), 1e-6, "Extrapolation above both failed");
    }

    @Test
    void testClampMode() {
        TwoDTabulatedFunction f = makeFunc(true);
        assertEquals(0f, f.evaluate(-10f, 0f), 1e-6, "Clamp below X failed");
        assertEquals(5f, f.evaluate(5f, -10f), 1e-6, "Clamp below Y failed");
        assertEquals(10f, f.evaluate(5f, 5f), 1e-6, "Clamp at top-right failed");
        assertEquals(10f, f.evaluate(100f, 100f), 1e-6, "Clamp far outside failed");
    }

    @Test
    void testHoles() {
        TwoDTabulatedFunction f = makeFuncWithHoles(false);

        // Hole in X row: x=2 missing → should interpolate between x=1 and x=3
        float val = f.evaluate(2f, 4f);
        assertEquals(6f, val, 1e-6, "Interpolation across missing row failed");

        // Hole in Y entry: y=3 missing → should interpolate between y=2 and y=4
        val = f.evaluate(4f, 3f);
        assertEquals(7f, val, 1e-6, "Interpolation across missing column entry failed");
    }

    @Test
    void stressRandomInputs() {
        TwoDTabulatedFunction f = makeFuncWithHoles(false);
        Random rnd = new Random(42);

        for (int i = 0; i < 5000; i++) {
            float x = rnd.nextFloat() * 7f - 1f; // [-1,6]
            float y = rnd.nextFloat() * 7f - 1f; // [-1,6]
            float expected = x + y; // true function
            float actual = f.evaluate(x, y);

            assertEquals(expected, actual, 1e-4,
                    "Stress test failed at (" + x + "," + y + ")");
        }
    }
}
