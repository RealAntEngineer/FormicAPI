package com.rae.formicapi.pde;

import com.rae.formicapi.foundation.math.pde.Field;
import com.rae.formicapi.foundation.math.pde.FieldType;
import com.rae.formicapi.foundation.math.pde.SymbolBinding;
import com.rae.formicapi.foundation.math.pde.SymbolRole;
import com.rae.formicapi.foundation.math.pde.ast.*;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class TestVectorTensorAlgebra {

    private static SymbolBinding scalar(String name, SymbolRole role) {
        return new SymbolBinding(new Field(name, FieldType.SCALAR), role);
    }

    private static SymbolBinding vector(String name, SymbolRole role) {
        return new SymbolBinding(new Field(name, FieldType.VECTOR), role);
    }

    private static final SymbolBinding T = scalar("T", SymbolRole.UNKNOWN);
    private static final SymbolBinding K = scalar("k", SymbolRole.COEFFICIENT);
    private static final SymbolBinding V = vector("V", SymbolRole.COEFFICIENT);

    @Test
    void gradOfScalarExpandsToVectorOfAxisDerivatives() {
        Expression expr = new UnaryExpression(UnaryOperators.GRAD, new VariableExpression(T));
        Expression result = ExpressionAlgebra.distribute(expr, 3);

        assertEquals(
                new VectorExpression(List.of(
                        new UnaryExpression(UnaryOperators.DDX, new VariableExpression(T)),
                        new UnaryExpression(UnaryOperators.DDY, new VariableExpression(T)),
                        new UnaryExpression(UnaryOperators.DDZ, new VariableExpression(T))
                )),
                result
        );
    }

    @Test
    void gradOfScalarRespectsLowerDimensionCount() {
        Expression expr = new UnaryExpression(UnaryOperators.GRAD, new VariableExpression(T));
        VectorExpression result = (VectorExpression) ExpressionAlgebra.distribute(expr, 2);
        assertEquals(2, result.dimension());
    }

    @Test
    void divOfVectorSymbolExpandsUsingComponentExpression() {
        Expression expr = new UnaryExpression(UnaryOperators.DIV, new VariableExpression(V));
        Expression result = ExpressionAlgebra.distribute(expr, 3);

        Expression expected = new BinaryExpression(BinaryOperators.ADD,
                new BinaryExpression(BinaryOperators.ADD,
                        new UnaryExpression(UnaryOperators.DDX, ComponentExpression.ofVector(new VariableExpression(V), 0)),
                        new UnaryExpression(UnaryOperators.DDY, ComponentExpression.ofVector(new VariableExpression(V), 1))),
                new UnaryExpression(UnaryOperators.DDZ, ComponentExpression.ofVector(new VariableExpression(V), 2)));

        assertEquals(expected, result);
    }

    @Test
    void divOfKTimesGradExpandsFullyIntoScalarAxisTerms() {
        // div(k*grad(T)) — regression test for the scalar*vector distribution fix:
        // grad(T) lowers to a VectorExpression, so k*VectorExpression must distribute
        // componentwise rather than leaving an opaque ComponentExpression wrapping
        // the whole product (which div would then wrap ddx/ddy/ddz around uselessly).
        Expression expr = new UnaryExpression(UnaryOperators.DIV,
                new BinaryExpression(BinaryOperators.MULTIPLY,
                        new VariableExpression(K),
                        new UnaryExpression(UnaryOperators.GRAD, new VariableExpression(T))));

        Expression result = ExpressionAlgebra.distribute(expr, 3);

        Expression kDdxT = new BinaryExpression(BinaryOperators.MULTIPLY, new VariableExpression(K), new UnaryExpression(UnaryOperators.DDX, new VariableExpression(T)));
        Expression kDdyT = new BinaryExpression(BinaryOperators.MULTIPLY, new VariableExpression(K), new UnaryExpression(UnaryOperators.DDY, new VariableExpression(T)));
        Expression kDdzT = new BinaryExpression(BinaryOperators.MULTIPLY, new VariableExpression(K), new UnaryExpression(UnaryOperators.DDZ, new VariableExpression(T)));

        Expression expected = new BinaryExpression(BinaryOperators.ADD,
                new BinaryExpression(BinaryOperators.ADD,
                        new UnaryExpression(UnaryOperators.DDX, kDdxT),
                        new UnaryExpression(UnaryOperators.DDY, kDdyT)),
                new UnaryExpression(UnaryOperators.DDZ, kDdzT));

        assertEquals(expected, result);
    }

    @Test
    void vectorPlusVectorLowersComponentwise() {
        SymbolBinding a = vector("a", SymbolRole.COEFFICIENT);
        SymbolBinding b = vector("b", SymbolRole.COEFFICIENT);

        Expression expr = new BinaryExpression(BinaryOperators.ADD, new VariableExpression(a), new VariableExpression(b));
        Expression result = ExpressionAlgebra.distribute(expr, 2);

        Expression expected = new VectorExpression(List.of(
                new BinaryExpression(BinaryOperators.ADD,
                        ComponentExpression.ofVector(new VariableExpression(a), 0),
                        ComponentExpression.ofVector(new VariableExpression(b), 0)),
                new BinaryExpression(BinaryOperators.ADD,
                        ComponentExpression.ofVector(new VariableExpression(a), 1),
                        ComponentExpression.ofVector(new VariableExpression(b), 1))
        ));

        assertEquals(expected, result);
    }

    @Test
    void dotProductOfVectorExpressionsExpandsToComponentSum() {
        VectorExpression a = new VectorExpression(List.of(new ConstantExpression(1), new ConstantExpression(2), new ConstantExpression(3)));
        VectorExpression b = new VectorExpression(List.of(new ConstantExpression(4), new ConstantExpression(5), new ConstantExpression(6)));

        Expression result = ExpressionAlgebra.distribute(new BinaryExpression(BinaryOperators.DOT_PRODUCT, a, b), 3);

        assertEquals("(((1.0 * 4.0) + (2.0 * 5.0)) + (3.0 * 6.0))", result.toString());
    }

    @Test
    void crossProductRequiresExactlyThreeDimensions() {
        VectorExpression a = new VectorExpression(List.of(new ConstantExpression(1), new ConstantExpression(0)));
        VectorExpression b = new VectorExpression(List.of(new ConstantExpression(0), new ConstantExpression(1)));

        assertThrows(UnsupportedOperationException.class,
                () -> ExpressionAlgebra.distribute(new BinaryExpression(BinaryOperators.CROSS_PRODUCT, a, b), 2));
    }

    @Test
    void outerProductProducesDimensionSquaredTensor() {
        VectorExpression a = new VectorExpression(List.of(new ConstantExpression(1), new ConstantExpression(2)));
        VectorExpression b = new VectorExpression(List.of(new ConstantExpression(3), new ConstantExpression(4)));

        TensorExpression result = (TensorExpression) ExpressionAlgebra.distribute(new BinaryExpression(BinaryOperators.OUTER_PRODUCT, a, b), 2);

        assertEquals(2, result.rowCount());
        assertEquals(2, result.columnCount());
    }

    @Test
    void dotProductTypeMismatchThrowsAtConstruction() {
        // resultType() is now validated eagerly in each record's compact
        // constructor, so a mismatch throws when the BinaryExpression itself
        // is built — not later, when something asks for its type.
        Expression scalarExpr = new VariableExpression(T);
        Expression vectorExpr = new VariableExpression(V);

        assertThrows(FieldType.TypeMismatchException.class,
                () -> new BinaryExpression(BinaryOperators.DOT_PRODUCT, scalarExpr, vectorExpr));
    }

    @Test
    void dotProductDistributesOverAddition() {
        // (a+b).c -> (a.c) + (b.c) — bilinearity, exercised through the scalar-only
        // distribute() pass; doesn't need the dimensions overload since it never
        // touches grad/div/lap or an actual product expansion.
        SymbolBinding a = vector("a", SymbolRole.COEFFICIENT);
        SymbolBinding b = vector("b", SymbolRole.COEFFICIENT);
        SymbolBinding c = vector("c", SymbolRole.COEFFICIENT);

        Expression expr = new BinaryExpression(BinaryOperators.DOT_PRODUCT,
                new BinaryExpression(BinaryOperators.ADD, new VariableExpression(a), new VariableExpression(b)),
                new VariableExpression(c));

        Expression result = ExpressionAlgebra.distribute(expr);

        assertEquals("((a . c) + (b . c))", result.toString());
    }
}
