package com.rae.formicapi.pde;

import com.rae.formicapi.foundation.math.pde.Field;
import com.rae.formicapi.foundation.math.pde.FieldType;
import com.rae.formicapi.foundation.math.pde.SymbolBinding;
import com.rae.formicapi.foundation.math.pde.SymbolRole;
import com.rae.formicapi.foundation.math.pde.ast.Expression;

import java.util.Map;

public class PDEUtil {

    public static SymbolBinding variable(String name, SymbolRole role) {
        return new SymbolBinding(new Field(name, FieldType.SCALAR), role);
    }

    public static SymbolBinding variable(String name) {
        return new SymbolBinding(new Field(name, FieldType.SCALAR), SymbolRole.COEFFICIENT);
    }

    static SymbolBinding vector(String name, SymbolRole role) {
        return new SymbolBinding(new Field(name, FieldType.VECTOR), role);
    }

    private static final Map<String, SymbolBinding> SYMBOLS = Map.of(
            "a", variable("a"),
            "b", variable("b"),
            "c", variable("c"),
            "d", variable("d"),
            "e1", vector("e1", SymbolRole.UNKNOWN),
            "e2",vector("e2", SymbolRole.UNKNOWN),
            "k", variable("k", SymbolRole.EVALUATED_FIELD),
            "T", variable("T", SymbolRole.UNKNOWN)
    );

    public static Expression parse(String expression) {
        return Expression.parseExpression(expression, SYMBOLS, 3,0);
    }
}
