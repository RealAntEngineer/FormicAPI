package com.rae.formicapi.pde;

import com.rae.formicapi.foundation.math.pde.stencil.TimeIntegrationScheme;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class TestTimeIntegrationScheme {

    @Test
    void explicitThetaIsZero() {
        assertEquals(0.0, TimeIntegrationScheme.EXPLICIT.theta());
    }

    @Test
    void implicitThetaIsOne() {
        assertEquals(1.0, TimeIntegrationScheme.IMPLICIT.theta());
    }

    @Test
    void crankNicolsonThetaIsHalf() {
        assertEquals(0.5, TimeIntegrationScheme.CRANK_NICOLSON.theta());
    }
}