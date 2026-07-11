package com.rae.formicapi.foundation.math.pde;


import com.rae.formicapi.foundation.math.pde.ast.*;

import java.util.*;

public final class Equation {

    private final Map<String, SymbolBinding> symbols = new LinkedHashMap<>();

    private final Expression left;
    private final Expression right;


    public Equation(String expression, SymbolBinding... fieldBindings) {
        for (SymbolBinding bind : fieldBindings) {
            SymbolBinding previous = symbols.put(bind.field().name(), bind);
            if (previous != null) {
                throw new IllegalArgumentException("Duplicate symbol: " + bind.field().name());
            }
        }
        expression = expression.replaceAll("\\s+", " ");
        String[] split = expression.split("=");

        if (split.length != 2)
            throw new RuntimeException("Failed to parse equation, wrong number of equality : "
                    + split.length + " for string" + expression);

        left = Expression.parseExpression(split[0], symbols,0);
        right = Expression.parseExpression(split[1], symbols, 0);
    }

    //split the text in 2 so that the



    public Expression getLeft() {
        return left;
    }

    public Expression getRight() {
        return right;
    }

    @Override
    public String toString() {
        return "Equation{" +Expression.print(left) +
                " = " + Expression.print(right) +
                ", symbols=" + Arrays.toString(symbols.values().toArray()) +
                '}';
    }
}