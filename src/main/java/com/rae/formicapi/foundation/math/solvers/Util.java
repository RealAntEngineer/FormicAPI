package com.rae.formicapi.foundation.math.solvers;

import com.rae.formicapi.foundation.math.operators.vectors.IntegerVector;

public class Util {

    //todo this isn't gpu friendly -> set should get a flush command in the gpu. or build an explicitly
    // cpu array before sending it back to the gpu. (migration commands ?)
    /**
     * Utility static function for constructing the unknownIdx for constrained solving of a "non-square" system.
     *
     * @param fixedVariables a pattern of fixed variables that are the input of an equation system, true = fixed, false = free.
     * @param unknownIdx a pointer that is the result of this function, gives a mapping of the free variables.
     * @param variableSize the size of the input variable of the system, can be larger than fixedVariables in the case of pattern recognizing
     * @param nbrOfEquations the number of equation the system possess, also named output size.
     */
    public static void fillUnknowIdx(boolean[] fixedVariables, IntegerVector unknownIdx, int variableSize, int nbrOfEquations) {
        int idx = 0;

        int patternSize = fixedVariables.length;

        if (variableSize % patternSize == 0) {
            for (int i = 0; i < variableSize; i++) {
                if (!fixedVariables[i % patternSize]) {
                    if (idx >= nbrOfEquations)
                        throw new IllegalArgumentException("requires number of free variables == equations");
                    unknownIdx.set(i, idx);
                    idx++;
                }
            }
        } else {
            throw new IllegalArgumentException("Impossible to recognise the pattern for a fixedVariables array of size "
                    + patternSize + " for a system with an input of size "+ variableSize);
        }
    }
}
