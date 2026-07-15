package com.rae.formicapi.foundation.math.pde.stencil;

import com.rae.formicapi.foundation.math.pde.FieldType;
import com.rae.formicapi.foundation.math.pde.ScalarAlgebra;
import com.rae.formicapi.foundation.math.pde.SymbolBinding;
import com.rae.formicapi.foundation.math.pde.ast.VariableExpression;

import java.util.Arrays;
import java.util.Objects;
import java.util.stream.Collectors;

/**
 * Represents a reference to a {@link SymbolBinding}'s value at a specific
 * point in the discretized space-time grid, relative to the node currently
 * being stamped.
 *
 * <p>This is the atomic leaf that differential operators (grad, div, lap,
 * ddt, d2dt) get lowered into during discretization: rather than an
 * {@code UnaryExpression(GRAD, ...)} node, a spatial derivative becomes an
 * ordinary arithmetic combination of {@code DiscretizedVariableExpression}
 * leaves at neighboring offsets, e.g. central differencing
 * {@code grad_x(T) ≈ (T[1,0,0] - T[-1,0,0]) / (2*dx)}.
 *
 * <p>{@code spatialOffset} has one entry per spatial dimension, so the same
 * type works for 1D, 2D, or 3D grids without change — a 1D field's offset is
 * {@code int[]{dx}}, a 2D field's is {@code int[]{dx,dy}}, and so on. All
 * instances referencing the same {@link SymbolBinding} in a given equation
 * are expected to share the same array length; nothing here enforces that
 * across an equation, so mixing dimensionalities for the same symbol is a
 * caller error.
 *
 * <ul>
 *     <li>{@code spatialOffset} — offset in grid cells per axis; all-zero is the central node.</li>
 *     <li>{@code dt} — temporal offset in time steps; {@code 0} is the unknown currently being
 *     solved for, negative values are known previous time levels (e.g. {@code -1} is last tick).</li>
 * </ul>
 *
 * <p>{@code equals}/{@code hashCode} are overridden by hand because records
 * do not generate structural comparisons for array components — the default
 * would compare {@code spatialOffset} by reference, silently breaking
 * {@link ScalarAlgebra}'s term-grouping (two independently built
 * references to the same offset would never merge).
 */
public class DiscretizedVariableExpression extends VariableExpression {
    private final int[] spatialOffset;
    private final int   dt;

    public DiscretizedVariableExpression(SymbolBinding name, int[] spatialOffset, int dt) {
        super(name, spatialOffset.length);
        this.spatialOffset = spatialOffset.clone();
        this.dt = dt;
    }

    public static DiscretizedVariableExpression atCentralNode(SymbolBinding symbol, int dimensions) {
        return new DiscretizedVariableExpression(symbol, new int[dimensions], 0);
    }

    public static DiscretizedVariableExpression atCentralNode(SymbolBinding symbol, int dimensions, int dt) {
        return new DiscretizedVariableExpression(symbol, new int[dimensions], dt);
    }

    /**
     * Returns a defensive copy — mutating the result does not affect this instance.
     */
    public int[] spatialOffset() {
        return spatialOffset.clone();
    }

    public int offset(int axis) {
        return spatialOffset[axis];
    }

    public DiscretizedVariableExpression withSpatialOffset(int... deltas) {
        if (deltas.length != spatialOffset.length) {
            throw new IllegalArgumentException(
                    "Expected " + spatialOffset.length + " offset component(s) for " + dimension() + "D symbol '"
                            + getSymbol().field().name() + "', got " + deltas.length);
        }
        int[] newOffset = new int[spatialOffset.length];
        for (int i = 0; i < spatialOffset.length; i++) {
            newOffset[i] = spatialOffset[i] + deltas[i];
        }
        return new DiscretizedVariableExpression(getSymbol(), newOffset, dt);
    }

    public int dimension() {
        return spatialOffset.length;
    }

    public DiscretizedVariableExpression withTemporalOffset(int ddt) {
        return new DiscretizedVariableExpression(getSymbol(), spatialOffset, dt + ddt);
    }

    public boolean isCurrentTimeLevel() {
        return dt == 0;
    }

    @Override
    public int hashCode() {
        return Objects.hash(getSymbol(), Arrays.hashCode(spatialOffset), dt);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof DiscretizedVariableExpression other)) return false;
        return dt == other.dt
                && getSymbol().equals(other.getSymbol())
                && Arrays.equals(spatialOffset, other.spatialOffset);
    }

    @Override
    public FieldType resultType() {
        return getSymbol().field().type();
    }

    @Override
    public String prettyPrint() {
        String result = getSymbol().field().name();

        result += Arrays.stream(spatialOffset)
                .mapToObj(Integer::toString)
                .collect(Collectors.joining(",", "[", "]"));


        if (dt != 0) {
            result += "{" + (dt > 0 ? "+" : "") + dt + "}";
        }

        return result;
    }

    public boolean isCentral() {
        for (int offset : spatialOffset) {
            if (offset != 0) return false;
        }
        return true;
    }

    @Override
    public boolean isTimeDifferentiable() {
        return false;//discretization already applied no ?
    }

    @Override
    public boolean isSpaceDifferentiable() {
        return false;
    }

    @Override
    public String toString() {
        return "DiscretizedVariableExpression{" +
                "spatialOffset=" + Arrays.toString(spatialOffset) +
                ", dt=" + dt +
                '}';
    }
}