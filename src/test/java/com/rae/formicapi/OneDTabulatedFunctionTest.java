package com.rae.formicapi;

import com.rae.formicapi.math.data.OneDTabulatedFunction;
import com.rae.formicapi.math.data.StepMode;
import org.junit.jupiter.api.Test;

import java.util.Random;
import java.util.TreeMap;

import static org.junit.jupiter.api.Assertions.*;

class OneDTabulatedFunctionTest {

    private OneDTabulatedFunction makeLinearFunc(boolean clamp) {
        // y = 2x, full table
        TreeMap<Float, Float> table = new TreeMap<>();
        for (int i = 0; i <= 10; i++) {
            table.put((float) i, (float) (2 * i));
        }
        return new OneDTabulatedFunction(table, 1f, StepMode.LINEAR, clamp);
    }

    private OneDTabulatedFunction makeLinearFuncWithHoles(boolean clamp) {
        // y = 2x, but missing some keys (holes at 3,4,7)
        TreeMap<Float, Float> table = new TreeMap<>();
        for (int i = 0; i <= 10; i++) {
            if (i == 3 || i == 4 || i == 7) continue; // hole
            table.put((float) i, (float) (2 * i));
        }
        return new OneDTabulatedFunction(table, 1f, StepMode.LINEAR, clamp);
    }

    @Test
    void testInterpolationExactPoints() {
        OneDTabulatedFunction f = makeLinearFunc(false);
        for (int i = 0; i <= 10; i++) {
            assertEquals(2f * i, f.evaluate(i), 1e-6,
                    "Interpolation at exact table point failed for x=" + i);
        }
    }

    @Test
    void testInterpolationMidpoints() {
        OneDTabulatedFunction f = makeLinearFunc(false);
        for (int i = 0; i < 10; i++) {
            float mid = i + 0.5f;
            float expected = 2f * mid;
            assertEquals(expected, f.evaluate(mid), 1e-6,
                    "Interpolation at midpoint failed for x=" + mid);
        }
    }

    @Test
    void testExtrapolationBelow() {
        OneDTabulatedFunction f = makeLinearFunc(false);
        assertEquals(-2f, f.evaluate(-1f), 1e-6, "Extrapolation below failed");
        assertEquals(-20f, f.evaluate(-10f), 1e-6, "Extrapolation far below failed");
    }

    @Test
    void testExtrapolationAbove() {
        OneDTabulatedFunction f = makeLinearFunc(false);
        assertEquals(22f, f.evaluate(11f), 1e-6, "Extrapolation above failed");
        assertEquals(40f, f.evaluate(20f), 1e-6, "Extrapolation far above failed");
    }

    @Test
    void testClampMode() {
        OneDTabulatedFunction f = makeLinearFunc(true);
        assertEquals(0f, f.evaluate(-10f), 1e-6, "Clamp below failed");
        assertEquals(20f, f.evaluate(10f), 1e-6, "Clamp at max key failed");
        assertEquals(20f, f.evaluate(15f), 1e-6, "Clamp above failed");
    }

    @Test
    void testInterpolationWithHoles() {
        OneDTabulatedFunction f = makeLinearFuncWithHoles(false);

        // Direct point that exists
        assertEquals(6f, f.evaluate(3f), 1e-6,
                "Should interpolate hole at x=3 using neighbors");

        assertEquals(8f, f.evaluate(4f), 1e-6,
                "Should interpolate hole at x=4 using neighbors");

        assertEquals(14f, f.evaluate(7f), 1e-6,
                "Should interpolate hole at x=7 using neighbors");

        // Midpoint across hole
        float expected = 2f * 3.5f; // linear
        assertEquals(expected, f.evaluate(3.5f), 1e-6,
                "Interpolation across hole failed");
    }

    @Test
    void stressRandomInputs() {
        OneDTabulatedFunction f = makeLinearFuncWithHoles(false);
        Random rnd = new Random(42);

        for (int i = 0; i < 10000; i++) {
            float x = rnd.nextFloat() * 40f - 10f; // Range [-10, 30]
            float expected = 2f * x; // true underlying function
            float actual = f.evaluate(x);

            assertEquals(expected, actual, 1,
                    "Stress test failed at x=" + x);
        }
    }
}
