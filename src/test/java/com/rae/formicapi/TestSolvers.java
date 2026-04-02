package com.rae.formicapi;

import com.rae.formicapi.math.Solvers;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TestSolvers {

    @Test
    public void testGradientDescent1() {
        float result = Solvers.gradientDecent((x)-> x*x,-1, 0.1f,0.01f);
        float tolerance = 1e-6f;
        assertTrue(result > 0- tolerance && result < 0 + tolerance);
    }
    @Test
    public void testGradientDescent2() {
        float result = Solvers.gradientDecent((x)-> (float) Math.abs(Math.cos(x)),0.001f, 0.1f,0.01f);
        float tolerance = 1e-3f;
        double expected = Math.PI/2f;
        assertEquals(expected,result, tolerance,() -> String.valueOf(result));
    }
}