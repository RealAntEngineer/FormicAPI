package com.rae.formicapi.pde;

import com.rae.formicapi.foundation.math.pde.*;
import com.rae.formicapi.foundation.math.pde.ast.*;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Vector/tensor-valued operators built directly as an AST (rather than
 * through the string parser): componentwise addition, dot/cross/outer
 * products, and type-mismatch validation.
 */
@SuppressWarnings("SameParameterValue")
public class TestVectorTensorAlgebra {

    private static SymbolBinding scalar(String name, SymbolRole role) {
        return new SymbolBinding(new Field(name, FieldType.SCALAR), role);
    }

    private static SymbolBinding vector(String name, SymbolRole role) {
        return new SymbolBinding(new Field(name, FieldType.VECTOR), role);
    }

    private static final SymbolBinding T = scalar("T", SymbolRole.UNKNOWN);
    private static final SymbolBinding V = vector("V", SymbolRole.COEFFICIENT);

    @Test
    void gradOfScalarExpandsToVectorOfAxisDerivatives() {
        Expression expr = new UnaryExpression(UnaryOperators.GRAD, new VariableExpression(T, 3));
        Expression result = expr.expand();

        assertEquals(
                new VectorExpression(List.of(
                        new UnaryExpression(UnaryOperators.DDX, new VariableExpression(T, 3)),
                        new UnaryExpression(UnaryOperators.DDY, new VariableExpression(T, 3)),
                        new UnaryExpression(UnaryOperators.DDZ, new VariableExpression(T, 3))
                )),
                result
        );
    }

    @Test
    void gradOfScalarRespectsLowerDimensionCount() {
        Expression expr = new UnaryExpression(UnaryOperators.GRAD, new VariableExpression(T, 2));
        VectorExpression result = (VectorExpression) expr.expand();
        assertEquals(2, result.dimension());
    }

    @Test
    void divOfVectorSymbolExpandsUsingComponentExpression() {
        Expression expr = new UnaryExpression(UnaryOperators.DIV, new VariableExpression(V, 3));
        Expression result = expr.expand();

        Expression expected = new BinaryExpression(BinaryOperators.ADD,
                new BinaryExpression(BinaryOperators.ADD,
                        new UnaryExpression(UnaryOperators.DDX, new VariableExpression(V, 3).componentAt(0)),
                        new UnaryExpression(UnaryOperators.DDY, new VariableExpression(V, 3).componentAt(1))
                ),
                new UnaryExpression(UnaryOperators.DDZ, new VariableExpression(V, 3).componentAt(2)));

        assertEquals(expected, result);
    }

    @Test
    void vectorPlusVectorLowersComponentwise() {
        SymbolBinding a = vector("a", SymbolRole.COEFFICIENT);
        SymbolBinding b = vector("b", SymbolRole.COEFFICIENT);

        Expression expr = new BinaryExpression(BinaryOperators.ADD, new VariableExpression(a, 2), new VariableExpression(b, 2));
        Expression result = expr.expand();

        Expression expected = new VectorExpression(List.of(
                new BinaryExpression(BinaryOperators.ADD,
                        ComponentExpression.ofVector(new VariableExpression(a, 2), 0),
                        ComponentExpression.ofVector(new VariableExpression(b, 2), 0)),
                new BinaryExpression(BinaryOperators.ADD,
                        ComponentExpression.ofVector(new VariableExpression(a, 2), 1),
                        ComponentExpression.ofVector(new VariableExpression(b, 2), 1))
        ));

        assertEquals(expected, result);
    }

    @Test
    void dotProductOfVectorExpressionsExpandsToComponentSum() {
        VectorExpression a = new VectorExpression(List.of(new ConstantExpression(1, 3), new ConstantExpression(2, 3), new ConstantExpression(3, 3)));
        VectorExpression b = new VectorExpression(List.of(new ConstantExpression(4, 3), new ConstantExpression(5, 3), new ConstantExpression(6, 3)));

        Expression result = new BinaryExpression(BinaryOperators.DOT_PRODUCT, a, b).expand();

        assertEquals("1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0", result.prettyPrint());
    }

    @Test
    void crossProductRequiresExactlyThreeDimensions() {
        VectorExpression a = new VectorExpression(List.of(new ConstantExpression(1, 2), new ConstantExpression(0, 2)));
        VectorExpression b = new VectorExpression(List.of(new ConstantExpression(0, 2), new ConstantExpression(1, 2)));

        assertThrows(UnsupportedOperationException.class,
                () -> new BinaryExpression(BinaryOperators.CROSS_PRODUCT, a, b).expand());
    }

    @Test
    void outerProductProducesDimensionSquaredTensor() {
        VectorExpression a = new VectorExpression(List.of(new ConstantExpression(1, 2), new ConstantExpression(2, 2)));
        VectorExpression b = new VectorExpression(List.of(new ConstantExpression(3, 2), new ConstantExpression(4, 2)));

        MatrixExpression result = (MatrixExpression) new BinaryExpression(BinaryOperators.OUTER_PRODUCT, a, b).expand();

        assertEquals(2, result.rowCount());
        assertEquals(2, result.columnCount());
    }

    @Test
    void dotProductTypeMismatchThrowsAtConstruction() {
        // resultType() is now validated eagerly in each record's compact
        // constructor, so a mismatch throws when the BinaryExpression itself
        // is built — not later, when something asks for its type.
        Expression scalarExpr = new VariableExpression(T, 3);
        Expression vectorExpr = new VariableExpression(V, 3);

        assertThrows(FieldType.TypeMismatchException.class,
                () -> new BinaryExpression(BinaryOperators.DOT_PRODUCT, scalarExpr, vectorExpr));
    }

    @Test
    void dotProductDistributesOverAdditionOfVectors() {
        // (a+b).c -> (a.c) + (b.c) — bilinearity, exercised through the scalar-only
        // distribute() pass; doesn't need the dimensions overload since it never
        // touches grad/div/lap or an actual product expansion.
        SymbolBinding a = vector("a", SymbolRole.COEFFICIENT);
        SymbolBinding b = vector("b", SymbolRole.COEFFICIENT);
        SymbolBinding c = vector("c", SymbolRole.COEFFICIENT);

        Expression expr = new BinaryExpression(BinaryOperators.DOT_PRODUCT,
                new BinaryExpression(BinaryOperators.ADD,
                        new VariableExpression(a, 3),
                        new VariableExpression(b, 3)),
                new VariableExpression(c, 3));

        assertEquals("(a + b) . c", expr.prettyPrint());

        Expression result = expr.expand().expand();

        assertEquals("a[0] * c[0] + b[0] * c[0] + a[1] * c[1] + b[1] * c[1] + a[2] * c[2] + b[2] * c[2]", result.prettyPrint());
    }
}