package com.rae.formicapi.foundation.math.operators.linear;

import com.rae.formicapi.foundation.math.operators.physics.Variable;
import com.rae.formicapi.foundation.math.operators.physics.VariableLayout;
import com.rae.formicapi.foundation.math.operators.physics.VariableMask;

import java.util.ArrayList;
import java.util.List;

/**
 * Block-CSR matrix over a mixed-variable state vector, where each block row/column
 * only spans the {@link Variable}s a given operator actually solves for.
 *
 * <p>Example: on a 5-variable state {@code (u, v, w, P, T)}, the URANS operator uses a
 * {@link VariableMask} of {@code (U, V, W, PRESSURE)} — every block is 4x4 and T never
 * appears. The Temperature operator uses a mask of {@code (TEMPERATURE)} alone — every
 * block is 1x1 and u, v, w, P never appear.
 *
 * <p>This mirrors CSR at the <b>cell</b> level, not the variable level:
 * <ul>
 *     <li>{@code rowPtr[cell]..rowPtr[cell+1]} bounds the blocks (neighbor cells) touched
 *     by {@code cell}'s equations — exactly like a CSR row pointer, except each entry is
 *     a dense {@code blockSize x blockSize} block instead of a scalar.</li>
 *     <li>{@code blockColumnCell[b]} is the neighbor cell for block {@code b}.</li>
 *     <li>{@code blockValues} stores each block's {@code blockSize*blockSize} values
 *     row-major, in the order given by {@code activeVariables.indices()}.</li>
 * </ul>
 *
 * <p>The variable mask is fixed for the whole matrix (decided once, at construction —
 * not per block, not per multiply), so {@code blockSize} is a real constant and the
 * inner multiply loop never branches on "is this variable active". Two small hand-written
 * kernels (1x1 and 4x4, the sizes URANS/Temperature actually need) cover the hot path;
 * anything else falls back to a plain generic block kernel.
 *
 * <p>{@code x} and {@code result} passed to {@link #multiply} are always the <b>full</b>
 * global state vector (size {@code nCells * layout.variablesPerCell()}). This matrix only
 * ever reads/writes the slots belonging to its own active variables — every other slot in
 * {@code result} is left untouched, so several operators (URANS, Temperature, ...) can
 * each write their own rows into a shared residual vector without stepping on each other.
 *
 * <p>Not thread-safe: {@link #multiply} and {@link #transposeMultiply} reuse internal
 * scratch buffers.
 */
@SuppressWarnings("unused")
public class BlockSparseMatrix implements MutableMatrix {

    private final int nCells;
    private final VariableLayout layout;
    private final VariableMask activeVariables;
    private final Variable[] localVariables; // size == blockSize, ascending ordinal order
    private final int blockSize;

    private final int[] rowPtr;           // size nCells + 1, cumulative block count per cell
    private final int[] blockColumnCell;  // size nnzBlocks, neighbor cell per block
    private final double[] blockValues;   // size nnzBlocks * blockSize * blockSize, row-major

    private final BlockKernel kernel;

    // reusable scratch — this is why the class is not thread-safe
    private final int[] rowGlobalIndexScratch;
    private final int[] colGlobalIndexScratch;
    private final double[] accumScratch;

    private BlockSparseMatrix(int nCells, VariableLayout layout, VariableMask activeVariables,
                              int[] rowPtr, int[] blockColumnCell) {
        this.nCells = nCells;
        this.layout = layout;
        this.activeVariables = activeVariables;
        this.localVariables = activeVariables.indices();
        this.blockSize = localVariables.length;
        this.rowPtr = rowPtr;
        this.blockColumnCell = blockColumnCell;
        this.blockValues = new double[blockColumnCell.length * blockSize * blockSize];
        this.kernel = BlockKernel.forSize(blockSize);

        this.rowGlobalIndexScratch = new int[blockSize];
        this.colGlobalIndexScratch = new int[blockSize];
        this.accumScratch = new double[blockSize];
    }

    // ------------------------------------------------------------------
    // Assembly
    // ------------------------------------------------------------------

    /**
     * Builds the fixed connectivity (which cells couple to which) up front, exactly like
     * building a CSR sparsity pattern. Values are filled in afterward via {@link #setBlock}
     * / {@link #addBlock} — no structural changes are possible once built.
     */
    public static class Builder {

        private final int nCells;
        private final VariableLayout layout;
        private final VariableMask activeVariables;
        private final List<List<Integer>> neighborsByCell;

        public Builder(int nCells, VariableLayout layout, VariableMask activeVariables) {
            this.nCells = nCells;
            this.layout = layout;
            this.activeVariables = activeVariables;
            this.neighborsByCell = new ArrayList<>(nCells);
            for (int i = 0; i < nCells; i++) {
                neighborsByCell.add(new ArrayList<>());
            }
        }

        /**
         * Declares that {@code cell}'s equations couple to {@code neighborCell} (use
         * {@code neighborCell == cell} for the diagonal — it is not added automatically).
         */
        public Builder addCoupling(int cell, int neighborCell) {
            neighborsByCell.get(cell).add(neighborCell);
            return this;
        }

        public BlockSparseMatrix build() {
            int[] rowPtr = new int[nCells + 1];
            int nnzBlocks = 0;
            for (int cell = 0; cell < nCells; cell++) {
                rowPtr[cell] = nnzBlocks;
                nnzBlocks += neighborsByCell.get(cell).size();
            }
            rowPtr[nCells] = nnzBlocks;

            int[] blockColumnCell = new int[nnzBlocks];
            int k = 0;
            for (int cell = 0; cell < nCells; cell++) {
                List<Integer> neighbors = neighborsByCell.get(cell);
                neighbors.sort(Integer::compareTo);
                for (int neighbor : neighbors) {
                    blockColumnCell[k++] = neighbor;
                }
            }

            return new BlockSparseMatrix(nCells, layout, activeVariables, rowPtr, blockColumnCell);
        }
    }

    // ------------------------------------------------------------------
    // Block-level access (preferred for assembly — O(1)/O(neighbors) instead of
    // translating global row/col indices back to cell+variable each time)
    // ------------------------------------------------------------------

    /** Index of the block coupling {@code cell} to {@code neighborCell}, or -1 if absent. */
    public int blockIndexOf(int cell, int neighborCell) {
        int start = rowPtr[cell];
        int end = rowPtr[cell + 1];
        for (int b = start; b < end; b++) {
            if (blockColumnCell[b] == neighborCell) {
                return b;
            }
        }
        return -1;
    }

    /**
     * Overwrites the block coupling {@code cell} to {@code neighborCell}.
     *
     * @param localValues row-major {@code blockSize x blockSize} values, ordered per
     *                    {@code activeVariables.indices()} (e.g. for URANS: U,V,W,PRESSURE)
     */
    public void setBlock(int cell, int neighborCell, double[] localValues) {
        setBlockAt(requireBlock(cell, neighborCell), localValues);
    }

    /** Same as {@link #setBlock} but by direct block index (fastest path for assembly loops). */
    public void setBlockAt(int blockIndex, double[] localValues) {
        checkBlockSize(localValues);
        System.arraycopy(localValues, 0, blockValues, blockIndex * blockSize * blockSize, blockSize * blockSize);
    }

    /** Accumulates into the block coupling {@code cell} to {@code neighborCell}. */
    public void addBlock(int cell, int neighborCell, double[] localValues) {
        addBlockAt(requireBlock(cell, neighborCell), localValues);
    }

    /** Same as {@link #addBlock} but by direct block index. */
    public void addBlockAt(int blockIndex, double[] localValues) {
        checkBlockSize(localValues);
        int base = blockIndex * blockSize * blockSize;
        for (int i = 0; i < localValues.length; i++) {
            blockValues[base + i] += localValues[i];
        }
    }

    private int requireBlock(int cell, int neighborCell) {
        int idx = blockIndexOf(cell, neighborCell);
        if (idx < 0) {
            throw new IllegalStateException(
                    "No block coupling cell " + cell + " to " + neighborCell +
                            " — connectivity is fixed at construction time");
        }
        return idx;
    }

    private void checkBlockSize(double[] localValues) {
        if (localValues.length != blockSize * blockSize) {
            throw new IllegalArgumentException(
                    "Expected " + (blockSize * blockSize) + " values (blockSize=" + blockSize +
                            ") but got " + localValues.length);
        }
    }

    // ------------------------------------------------------------------
    // Element-level access (MutableMatrix contract) — translates global (row, col)
    // back to (cell, variable) each call. Prefer the block-level methods above for assembly.
    // ------------------------------------------------------------------

    @Override
    public void set(int row, int col, double value) {
        int[] loc = locate(row, col);
        int blockIndex = requireBlock(loc[0], loc[2]);
        blockValues[blockIndex * blockSize * blockSize + loc[1] * blockSize + loc[3]] = value;
    }

    @Override
    public void add(int row, int col, double value) {
        if (value == 0.0) return;
        int[] loc = locate(row, col);
        int blockIndex = requireBlock(loc[0], loc[2]);
        blockValues[blockIndex * blockSize * blockSize + loc[1] * blockSize + loc[3]] += value;
    }

    @Override
    public double get(int row, int col) {
        int variablesPerCell = layout.variablesPerCell();
        int cellRow = row / variablesPerCell;
        int cellCol = col / variablesPerCell;
        Variable varRow = Variable.values()[row % variablesPerCell];
        Variable varCol = Variable.values()[col % variablesPerCell];

        if (!activeVariables.contains(varRow) || !activeVariables.contains(varCol)) {
            return 0.0;
        }

        int blockIndex = blockIndexOf(cellRow, cellCol);
        if (blockIndex < 0) {
            return 0.0;
        }

        int i = localIndexOf(varRow);
        int j = localIndexOf(varCol);
        return blockValues[blockIndex * blockSize * blockSize + i * blockSize + j];
    }

    /** @return {cellRow, localRow, cellCol, localCol}; throws if either variable is not active. */
    private int[] locate(int row, int col) {
        int variablesPerCell = layout.variablesPerCell();
        int cellRow = row / variablesPerCell;
        int cellCol = col / variablesPerCell;
        Variable varRow = Variable.values()[row % variablesPerCell];
        Variable varCol = Variable.values()[col % variablesPerCell];

        int i = localIndexOf(varRow);
        int j = localIndexOf(varCol);
        if (i < 0 || j < 0) {
            throw new IllegalStateException(
                    "(" + row + "," + col + ") touches a variable outside this operator's mask (" +
                            activeVariables + ")");
        }
        return new int[]{cellRow, i, cellCol, j};
    }

    private int localIndexOf(Variable v) {
        for (int i = 0; i < blockSize; i++) {
            if (localVariables[i] == v) return i;
        }
        return -1;
    }

    // ------------------------------------------------------------------
    // Multiply
    // ------------------------------------------------------------------

    @Override
    public void multiply(double[] x, double[] result) {
        int[] rowIdx = rowGlobalIndexScratch;
        int[] colIdx = colGlobalIndexScratch;
        double[] acc = accumScratch;

        for (int cell = 0; cell < nCells; cell++) {
            for (int k = 0; k < blockSize; k++) {
                rowIdx[k] = layout.index(cell, localVariables[k]);
                acc[k] = 0.0;
            }

            int start = rowPtr[cell];
            int end = rowPtr[cell + 1];

            for (int b = start; b < end; b++) {
                int neighborCell = blockColumnCell[b];
                for (int k = 0; k < blockSize; k++) {
                    colIdx[k] = layout.index(neighborCell, localVariables[k]);
                }
                kernel.multiplyAccumulate(blockValues, b * blockSize * blockSize, x, colIdx, acc);
            }

            for (int k = 0; k < blockSize; k++) {
                result[rowIdx[k]] = acc[k];
            }
        }
    }

    /**
     * Transpose multiply. Less hot than {@link #multiply} (adjoint/preconditioner use, not
     * the main residual path), so it uses a plain generic loop rather than the specialized
     * kernels. Only zeroes/writes the global indices this matrix owns, same as multiply.
     */
    @Override
    public void transposeMultiply(double[] x, double[] result) {
        for (int cell = 0; cell < nCells; cell++) {
            for (int k = 0; k < blockSize; k++) {
                result[layout.index(cell, localVariables[k])] = 0.0;
            }
        }

        int[] rowIdx = rowGlobalIndexScratch;

        for (int cell = 0; cell < nCells; cell++) {
            for (int k = 0; k < blockSize; k++) {
                rowIdx[k] = layout.index(cell, localVariables[k]);
            }

            int start = rowPtr[cell];
            int end = rowPtr[cell + 1];

            for (int b = start; b < end; b++) {
                int neighborCell = blockColumnCell[b];
                int blockOffset = b * blockSize * blockSize;

                for (int j = 0; j < blockSize; j++) {
                    int colGlobal = layout.index(neighborCell, localVariables[j]);
                    double sum = 0.0;
                    for (int i = 0; i < blockSize; i++) {
                        sum += blockValues[blockOffset + i * blockSize + j] * x[rowIdx[i]];
                    }
                    result[colGlobal] += sum;
                }
            }
        }
    }

    @Override
    public int rows() {
        return nCells * layout.variablesPerCell();
    }

    @Override
    public int cols() {
        return nCells * layout.variablesPerCell();
    }

    public VariableMask activeVariables() {
        return activeVariables;
    }

    public int blockSize() {
        return blockSize;
    }

    // ------------------------------------------------------------------
    // Kernels — chosen once at construction, never branched on per-multiply-call.
    // Only the sizes URANS (4) and Temperature (1) actually use are hand-unrolled;
    // anything else falls back to a plain generic block kernel.
    // ------------------------------------------------------------------

    @FunctionalInterface
    private interface BlockKernel {

        void multiplyAccumulate(double[] blockValues, int blockOffset, double[] x, int[] colIdx, double[] acc);

        static BlockKernel forSize(int blockSize) {
            switch (blockSize) {
                case 1:
                    return BlockKernel::multiply1;
                case 4:
                    return BlockKernel::multiply4;
                default:
                    return new GenericKernel(blockSize);
            }
        }

        static void multiply1(double[] blockValues, int blockOffset, double[] x, int[] colIdx, double[] acc) {
            acc[0] += blockValues[blockOffset] * x[colIdx[0]];
        }

        static void multiply4(double[] blockValues, int blockOffset, double[] x, int[] colIdx, double[] acc) {
            double x0 = x[colIdx[0]];
            double x1 = x[colIdx[1]];
            double x2 = x[colIdx[2]];
            double x3 = x[colIdx[3]];

            acc[0] += blockValues[blockOffset] * x0 + blockValues[blockOffset + 1] * x1
                    + blockValues[blockOffset + 2] * x2 + blockValues[blockOffset + 3] * x3;
            acc[1] += blockValues[blockOffset + 4] * x0 + blockValues[blockOffset + 5] * x1
                    + blockValues[blockOffset + 6] * x2 + blockValues[blockOffset + 7] * x3;
            acc[2] += blockValues[blockOffset + 8] * x0 + blockValues[blockOffset + 9] * x1
                    + blockValues[blockOffset + 10] * x2 + blockValues[blockOffset + 11] * x3;
            acc[3] += blockValues[blockOffset + 12] * x0 + blockValues[blockOffset + 13] * x1
                    + blockValues[blockOffset + 14] * x2 + blockValues[blockOffset + 15] * x3;
        }
    }

    private static final class GenericKernel implements BlockKernel {
        private final int n;

        GenericKernel(int n) {
            this.n = n;
        }

        @Override
        public void multiplyAccumulate(double[] blockValues, int blockOffset, double[] x, int[] colIdx, double[] acc) {
            for (int i = 0; i < n; i++) {
                double sum = 0.0;
                int rowOffset = blockOffset + i * n;
                for (int j = 0; j < n; j++) {
                    sum += blockValues[rowOffset + j] * x[colIdx[j]];
                }
                acc[i] += sum;
            }
        }
    }
}