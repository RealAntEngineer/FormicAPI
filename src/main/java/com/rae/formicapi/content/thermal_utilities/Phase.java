package com.rae.formicapi.content.thermal_utilities;

public enum Phase {
    LIQUID, SOLID, VAPOR,//no mix
    SUPERCRITICAL_VAPOR,
    SUPERCRITICAL_LIQUID,
    SUPERCRITICAL;


    public record PhaseMix(Phase[] phases, float[] fractions) {

        public PhaseMix(Phase phase) {
            this(new Phase[]{phase}, new float[]{1f});
        }
    }
}