package com.rae.formicapi.foundation.math.pde;

public enum FieldType {

    SCALAR,
    VECTOR,
    TENSOR;

    public static class TypeMismatchException extends RuntimeException {
        public TypeMismatchException(String message) {
            super(message);
        }
    }
}
