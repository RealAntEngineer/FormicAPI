package com.rae.formicapi.foundation.math.pde;

public enum FieldType {

    SCALAR,
    VECTOR,
    MATRIX;

    public static class TypeMismatchException extends RuntimeException {
        public TypeMismatchException(String message) {
            super(message);
        }
    }
}
