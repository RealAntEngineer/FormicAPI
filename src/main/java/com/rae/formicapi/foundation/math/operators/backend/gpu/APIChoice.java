package com.rae.formicapi.foundation.math.operators.backend.gpu;

import org.jetbrains.annotations.Nullable;

public class APIChoice {

    protected static @Nullable ImplementedAPI MODE = null;


    /**
     * Choose an API to base the Gpu compute shaders upon
     * @param api the api of choice
     * @return true if mode change successful
     */
    public static synchronized boolean setAPI(ImplementedAPI api){

        MODE = api;
        return true;
    }

    public enum ImplementedAPI {
        OpenCL
    }
}
