package com.rae.formicapi.foundation.math.pde;

public record Field(String name, FieldType type) {

    @Override
    public String toString() {
        return "Field{" +
                "name='" + name + '\'' +
                ", type=" + type +
                '}';
    }
}
