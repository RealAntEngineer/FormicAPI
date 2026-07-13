package com.rae.formicapi.foundation.math.pde.stencil;

import java.util.List;

/**
 * Selects which neighboring grid points a spatial derivative's finite-difference
 * approximation draws from, relative to the node being stamped.
 *
 * <p>Each scheme produces a {@link StencilPoint} list for a requested derivative
 * order along a single axis. Offsets are in grid cells (unitless); the caller is
 * responsible for dividing by the appropriate power of the grid spacing {@code h}
 * for that axis — this class only knows about stencil shape, not physical spacing.
 *
 * <pre>
 * CENTRAL,  1st order: (U[+1] - U[-1]) / (2h)
 * FORWARD,  1st order: (U[+1] - U[0])  / h
 * BACKWARD, 1st order: (U[0]  - U[-1]) / h
 * CENTRAL,  2nd order: (U[+1] - 2*U[0] + U[-1]) / h^2
 * </pre>
 */
public enum SpatialDifferenceScheme {

    CENTRAL {
        @Override
        public List<StencilPoint> stencil(int order) {
            return switch (order) {
                case 1 -> List.of(new StencilPoint(-1, -0.5), new StencilPoint(1, 0.5));
                case 2 -> List.of(new StencilPoint(-1, 1.0), new StencilPoint(0, -2.0), new StencilPoint(1, 1.0));
                default ->
                        throw new UnsupportedOperationException("CENTRAL scheme does not support derivative order " + order);
            };
        }
    },

    FORWARD {
        @Override
        public List<StencilPoint> stencil(int order) {
            return switch (order) {
                case 1 -> List.of(new StencilPoint(0, -1.0), new StencilPoint(1, 1.0));
                case 2 -> List.of(new StencilPoint(0, 1.0), new StencilPoint(1, -2.0), new StencilPoint(2, 1.0));
                default ->
                        throw new UnsupportedOperationException("FORWARD scheme does not support derivative order " + order);
            };
        }
    },

    BACKWARD {
        @Override
        public List<StencilPoint> stencil(int order) {
            return switch (order) {
                case 1 -> List.of(new StencilPoint(-1, -1.0), new StencilPoint(0, 1.0));
                case 2 -> List.of(new StencilPoint(-2, 1.0), new StencilPoint(-1, -2.0), new StencilPoint(0, 1.0));
                default ->
                        throw new UnsupportedOperationException("BACKWARD scheme does not support derivative order " + order);
            };
        }
    };

    /**
     * @param order derivative order (1 = gradient component, 2 = second derivative
     *              for a Laplacian component)
     * @return stencil points; weights are unitless and must be divided by
     * {@code h^order} by the caller, where {@code h} is the grid spacing along
     * the axis this stencil is being applied to
     */
    public abstract List<StencilPoint> stencil(int order);

    public record StencilPoint(int offset, double weight) {
    }
}
