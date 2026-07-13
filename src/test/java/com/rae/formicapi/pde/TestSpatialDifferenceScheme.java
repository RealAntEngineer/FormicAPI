package com.rae.formicapi.pde;

import com.rae.formicapi.foundation.math.pde.stencil.SpatialDifferenceScheme;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class TestSpatialDifferenceScheme {

    @Test
    void centralFirstOrderStencil() {
        List<SpatialDifferenceScheme.StencilPoint> stencil = SpatialDifferenceScheme.CENTRAL.stencil(1);
        assertEquals(List.of(new SpatialDifferenceScheme.StencilPoint(-1, -0.5), new SpatialDifferenceScheme.StencilPoint(1, 0.5)), stencil);
    }

    @Test
    void forwardFirstOrderStencil() {
        List<SpatialDifferenceScheme.StencilPoint> stencil = SpatialDifferenceScheme.FORWARD.stencil(1);
        assertEquals(List.of(new SpatialDifferenceScheme.StencilPoint(0, -1.0), new SpatialDifferenceScheme.StencilPoint(1, 1.0)), stencil);
    }

    @Test
    void backwardFirstOrderStencil() {
        List<SpatialDifferenceScheme.StencilPoint> stencil = SpatialDifferenceScheme.BACKWARD.stencil(1);
        assertEquals(List.of(new SpatialDifferenceScheme.StencilPoint(-1, -1.0), new SpatialDifferenceScheme.StencilPoint(0, 1.0)), stencil);
    }

    @Test
    void centralSecondOrderStencil() {
        List<SpatialDifferenceScheme.StencilPoint> stencil = SpatialDifferenceScheme.CENTRAL.stencil(2);
        assertEquals(List.of(new SpatialDifferenceScheme.StencilPoint(-1, 1.0), new SpatialDifferenceScheme.StencilPoint(0, -2.0), new SpatialDifferenceScheme.StencilPoint(1, 1.0)), stencil);
    }

    @Test
    void everyStencilSumsToZeroWeight() {
        // A finite-difference stencil approximating any derivative order >= 1
        // must have weights summing to zero: a constant field has zero derivative.
        for (SpatialDifferenceScheme scheme : SpatialDifferenceScheme.values()) {
            for (int order : List.of(1, 2)) {
                double sum = scheme.stencil(order).stream().mapToDouble(SpatialDifferenceScheme.StencilPoint::weight).sum();
                assertEquals(0.0, sum, 1e-12, scheme + " order " + order + " should sum to zero");
            }
        }
    }

    @Test
    void unsupportedOrderThrows() {
        assertThrows(UnsupportedOperationException.class, () -> SpatialDifferenceScheme.CENTRAL.stencil(3));
    }
}