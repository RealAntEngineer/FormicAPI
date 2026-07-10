package com.rae.formicapi.foundation.math.pde;

public record SymbolBinding(
        Field field,
        SymbolRole role
) {

    @Override
    public String toString() {
        return "SymbolBinding{" +
                "field=" + field +
                ", role=" + role +
                '}';
    }
}
