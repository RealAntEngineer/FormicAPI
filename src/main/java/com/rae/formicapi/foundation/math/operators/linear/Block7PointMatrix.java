package com.rae.formicapi.foundation.math.operators.linear;

import java.util.Arrays;

/**
 * Mutable block stencil matrix with a fixed sparsity structure.
 *
 * <p>This matrix represents a 3D 7-point stencil over blocks of
 * {@value #BLOCK_SIZE} elements. Each row contains exactly seven possible
 * coefficients:
 *
 * <ul>
 *     <li>center</li>
 *     <li>-X neighbor</li>
 *     <li>+X neighbor</li>
 *     <li>-Y neighbor</li>
 *     <li>+Y neighbor</li>
 *     <li>-Z neighbor</li>
 *     <li>+Z neighbor</li>
 * </ul>
 *
 * <p>The local stencil connectivity is implicit and determined by the voxel
 * position inside the block. Only block-to-block connectivity is stored
 * explicitly.
 *
 * <p>The sparsity pattern is fixed:
 * <ul>
 *     <li>No insertion or removal of entries is supported</li>
 *     <li>Only numerical values can be modified</li>
 *     <li>Missing neighboring blocks are represented by {@code -1}</li>
 * </ul>
 *
 * <p>This representation is optimized for structured problems such as:
 * <ul>
 *     <li>finite difference solvers</li>
 *     <li>finite element systems with regular connectivity</li>
 *     <li>diffusion and heat transfer simulations</li>
 *     <li>iterative linear solvers (CG, GMRES, etc.)</li>
 * </ul>
 */
@SuppressWarnings("unused")
public class Block7PointMatrix implements MutableMatrix {

    static final         byte   MIN_X         = 1;
    static final         byte   MAX_X         = 1 << 1;
    static final         byte   MIN_Y         = 1 << 2;
    static final         byte   MAX_Y         = 1 << 3;
    static final         byte   MIN_Z         = 1 << 4;
    static final         byte   MAX_Z         = 1 << 5;
    private static final int    BASE_BITS     = 4;
    private static final int    BASE_SIZE     = 1 << BASE_BITS;
    private static final int    X_WRAP        = BASE_SIZE - 1;
    private static final int    Z_WRAP        = X_WRAP * BASE_SIZE;
    private static final int    Y_WRAP        = X_WRAP * BASE_SIZE * BASE_SIZE;
    private static final int    Z_STEP        = BASE_SIZE;
    private static final int    Y_STEP        = BASE_SIZE * BASE_SIZE;
    private static final int    BLOCK_SIZE    = BASE_SIZE * BASE_SIZE * BASE_SIZE;
    static final         byte[] BOUNDARY_MASK = new byte[BLOCK_SIZE];
    private static final int    SELF          = 0;
    private static final int    XM            = 1;
    private static final int    XP            = 2;
    private static final int    YM            = 3;
    private static final int    YP            = 4;
    private static final int    ZM            = 5;
    private static final int    ZP            = 6;
    private static final int    NNZ_PER_ROW = 7;
    private static final int[] INTERIOR_IDX;
    private static final int[] BOUNDARY_IDX;

    static {

        for (int y = 0; y < BASE_SIZE; y++) {
            for (int z = 0; z < BASE_SIZE; z++) {
                for (int x = 0; x < BASE_SIZE; x++) {

                    int i = x | (z << BASE_BITS) | (y << BASE_BITS * 2);

                    byte mask = 0;

                    if (x == 0) mask |= MIN_X;
                    if (x == BASE_SIZE - 1) mask |= MAX_X;

                    if (y == 0) mask |= MIN_Y;
                    if (y == BASE_SIZE - 1) mask |= MAX_Y;

                    if (z == 0) mask |= MIN_Z;
                    if (z == BASE_SIZE - 1) mask |= MAX_Z;

                    BOUNDARY_MASK[i] = mask;
                }
            }
        }


        int interiorCount = 0;
        int boundaryCount = 0;

        for (int i = 0; i < BLOCK_SIZE; i++) {
            if (BOUNDARY_MASK[i] == 0) {
                interiorCount++;
            } else {
                boundaryCount++;
            }
        }


        INTERIOR_IDX = new int[interiorCount];
        BOUNDARY_IDX = new int[boundaryCount];


        int interiorPtr = 0;
        int boundaryPtr = 0;

        for (int i = 0; i < BLOCK_SIZE; i++) {

            if (BOUNDARY_MASK[i] == 0) {
                INTERIOR_IDX[interiorPtr++] = i;
            } else {
                BOUNDARY_IDX[boundaryPtr++] = i;
            }
        }
    }

    private              int blocks;

    private double[] values;

    private int[] blockPtr;


    /**
     * Constructs a fully defined CSR matrix.
     *
     * <p>The structure MUST already be consistent:
     * <ul>
     *     <li>{@code values.length == colIndex.length == nnz}</li>
     *     <li>{@code rowPtr.length == rows + 1}</li>
     *     <li>{@code rowPtr[rows] == nnz}</li>
     * </ul>
     *
     * <p>This constructor builds the fast lookup table used for updates.
     *
     * @param blocks : the number of BLOCK_SIZE blocks that are in this matrix
     *
     */
    public Block7PointMatrix(int blocks) {

        this.blocks = blocks;

        this.values = new double[blocks * BLOCK_SIZE * 7];
        this.blockPtr = new int[blocks * 6];

        Arrays.fill(blockPtr, -1);
    }

    /**
     * Resizes this matrix by changing the number of blocks.
     *
     * <p>If the matrix grows, newly allocated coefficients are initialized to
     * zero and new block neighbor pointers are initialized to {@code -1}.
     *
     * <p>If the matrix shrinks, trailing blocks are discarded logically.
     *
     * @param newBlocks new number of blocks
     */
    public void resize(int newBlocks) {

        int requiredValues   = newBlocks * BLOCK_SIZE * NNZ_PER_ROW;
        int requiredPointers = newBlocks * 6;

        if (requiredValues > values.length) {
            values = Arrays.copyOf(values, requiredValues);
        }

        if (requiredPointers > blockPtr.length) {

            int oldLength = blockPtr.length;

            blockPtr = Arrays.copyOf(blockPtr, requiredPointers);

            Arrays.fill(blockPtr, oldLength, requiredPointers, -1);
        }

        blocks = newBlocks;
    }

    /**
     * Adds a value to an existing entry in the CSR structure.
     *
     * <p>This method only works if the (row, col) entry already exists.
     * If it does not exist, an {@link IllegalStateException} is thrown.
     *
     * <p>Complexity: O(nnz_per_row) worst case (typically ~7).
     *
     * @param row   row index
     * @param col   column index
     * @param value value to add
     */
    @Override
    public void add(int row, int col, double value) {
        if (value == 0.0) return;
        //just throw if it's used.

        int block = row >> BASE_BITS * 3;
        int idx   = row & (BLOCK_SIZE - 1);

        int slot = findStencilSlot(block, idx, col);
        values[row * 7 + slot] += value;
    }

    /**
     * Sets a value in an existing CSR entry.
     *
     * <p>If the value is 0, the method simply writes 0 but does not remove
     * the entry (structure is fixed).
     *
     * <p>If the (row, col) entry does not exist, an exception is thrown.
     *
     * @param row   row index
     * @param col   column index
     * @param value new value
     */
    @Override
    public void set(int row, int col, double value) {

        int block = row >> BASE_BITS * 3;
        int idx   = row & (BLOCK_SIZE - 1);

        int slot = findStencilSlot(block, idx, col);//this doesn't work at all
        values[row * 7 + slot] = value;
    }

    private int findStencilSlot(int block, int idx, int col) {

        int ptr = block * 6;

        int self = block * BLOCK_SIZE;

        int row = self + idx;

        if (col == row)
            return 0;

        byte mask = BOUNDARY_MASK[idx];

        // -X
        if ((mask & MIN_X) == 0) {
            if (col == row - 1)
                return XM;
        } else if (blockPtr[ptr + XM - 1] != -1) {
            if (col == blockPtr[ptr + XM - 1] + idx + BASE_SIZE - 1)
                return XM;
        }

        // +X
        if ((mask & MAX_X) == 0) {
            if (col == row + 1)
                return XP;
        } else if (blockPtr[ptr + XP - 1] != -1) {
            if (col == blockPtr[ptr + XP - 1] + idx - (BASE_SIZE - 1))
                return XP;
        }

        // -Y
        if ((mask & MIN_Y) == 0) {
            if (col == row - BASE_SIZE * BASE_SIZE)
                return YM;
        } else if (blockPtr[ptr + YM - 1] != -1) {
            if (col == blockPtr[ptr + YM - 1] + idx + Y_WRAP)
                return YM;
        }

        // +Y
        if ((mask & MAX_Y) == 0) {
            if (col == row + BASE_SIZE * BASE_SIZE)
                return YP;
        } else if (blockPtr[ptr + YP - 1] != -1) {
            if (col == blockPtr[ptr + YP - 1] + idx - Y_WRAP)
                return YP;
        }

        // -Z
        if ((mask & MIN_Z) == 0) {
            if (col == row - BASE_SIZE)
                return ZM;
        } else if (blockPtr[ptr + ZM - 1] != -1) {
            if (col == blockPtr[ptr + ZM - 1] + idx + Z_WRAP)
                return ZM;
        }

        // +Z
        if ((mask & MAX_Z) == 0) {
            if (col == row + BASE_SIZE)
                return ZP;
        } else if (blockPtr[ptr + ZP - 1] != -1) {
            if (col == blockPtr[ptr + ZP - 1] + idx - Z_WRAP)
                return ZP;
        }

        throw new IllegalStateException(
                "Stencil entry does not exist: (" + row + "," + col + ")"
        );
    }

    /**
     * Replaces all seven stencil coefficients of a row.
     *
     * <p>The provided array corresponds to the fixed stencil ordering:
     *
     * <pre>
     * [SELF, XM, XP, YM, YP, ZM, ZP]
     * </pre>
     *
     * <p>If fewer than seven coefficients are provided, the remaining entries are
     * set to zero.
     *
     * @param row       row index to modify
     * @param newValues new stencil coefficients
     * @throws IllegalArgumentException if {@code newValues.length > 7}
     */
    public void setRow(int row, double[] newValues) {

        if (newValues.length > NNZ_PER_ROW) {
            throw new IllegalArgumentException(
                    "Expected arrays of size " + NNZ_PER_ROW +
                            " but got values=" + newValues.length
            );
        }

        int base = row * NNZ_PER_ROW;
        System.arraycopy(newValues, 0, values, base, NNZ_PER_ROW);
    }

    /**
     * Sets the neighboring blocks of a block.
     *
     * <p>The neighbor array must use the following ordering:
     *
     * <pre>
     * [XM, XP, YM, YP, ZM, ZP]
     * </pre>
     *
     * <p>A value of {@code -1} indicates that the corresponding neighboring block
     * does not exist.
     *
     * @param block          block index
     * @param neighborsBlock neighboring block start indices
     * @throws IllegalArgumentException if {@code neighborsBlock.length != 6}
     */
    public void setNeighbors(int block, int[] neighborsBlock) {
        if (neighborsBlock.length != 6) {
            throw new IllegalArgumentException("Expected 6 neighbors");
        }

        System.arraycopy(neighborsBlock, 0, blockPtr, block * 6, 6);
    }

    @Override
    public void multiply(double[] x, double[] result) {
        for (int block = 0; block < blocks; block++) {

            int ptr = block * 6;

            int self = block * BLOCK_SIZE;
            int xm   = blockPtr[ptr + XM - 1];
            int xp   = blockPtr[ptr + XP - 1];
            int ym   = blockPtr[ptr + YM - 1];
            int yp   = blockPtr[ptr + YP - 1];
            int zm   = blockPtr[ptr + ZM - 1];
            int zp   = blockPtr[ptr + ZP - 1];

            for (int idx = 0; idx < BLOCK_SIZE; idx++) {

                int row  = self + idx;
                int base = (self + idx) * 7;

                byte mask = BOUNDARY_MASK[idx];

                double sum = values[base] * x[row];

                // -X
                if ((mask & MIN_X) == 0) {
                    sum += values[base + XM] * x[row - 1];
                } else if (xm != -1) {
                    sum += values[base + XM] * x[xm + idx + BASE_SIZE - 1];
                }

                // +X
                if ((mask & MAX_X) == 0) {
                    sum += values[base + XP] * x[row + 1];
                } else if (xp != -1) {
                    sum += values[base + XP] * x[xp + idx - (BASE_SIZE - 1)];
                }

                // -Y
                if ((mask & MIN_Y) == 0) {
                    sum += values[base + YM] * x[row - BASE_SIZE * BASE_SIZE];
                } else if (ym != -1) {
                    sum += values[base + YM] * x[ym + idx + Y_WRAP];
                }

                // +Y
                if ((mask & MAX_Y) == 0) {
                    sum += values[base + YP] * x[row + BASE_SIZE * BASE_SIZE];
                } else if (yp != -1) {
                    sum += values[base + YP] * x[yp + idx - Y_WRAP];
                }

                // -Z
                if ((mask & MIN_Z) == 0) {
                    sum += values[base + ZM] * x[row - BASE_SIZE];
                } else if (zm != -1) {
                    sum += values[base + ZM] * x[zm + idx + Z_WRAP];
                }

                // +Z
                if ((mask & MAX_Z) == 0) {
                    sum += values[base + ZP] * x[row + BASE_SIZE];
                } else if (zp != -1) {
                    sum += values[base + ZP] * x[zp + idx - Z_WRAP];
                }

                result[row] = sum;
            }
        }
    }

    @Override
    public void transposeMultiply(double[] x, double[] result) {
        Arrays.fill(result, 0.0);

        for (int block = 0; block < blocks; block++) {

            int ptr = block * 6;

            int self = block * BLOCK_SIZE;
            int xm   = blockPtr[ptr + XM - 1];
            int xp   = blockPtr[ptr + XP - 1];
            int ym   = blockPtr[ptr + YM - 1];
            int yp   = blockPtr[ptr + YP - 1];
            int zm   = blockPtr[ptr + ZM - 1];
            int zp   = blockPtr[ptr + ZP - 1];

            for (int idx = 0; idx < BLOCK_SIZE; idx++) {

                int row  = self + idx;
                int base = row * 7;

                byte mask = BOUNDARY_MASK[idx];

                double xi = x[row];

                // diagonal
                result[row] += values[base] * xi;

                // -X
                if ((mask & MIN_X) == 0) {
                    result[row - 1] += values[base + XM] * xi;
                } else if (xm != -1) {
                    result[xm + idx + BASE_SIZE - 1] += values[base + XM] * xi;
                }

                // +X
                if ((mask & MAX_X) == 0) {
                    result[row + 1] += values[base + XP] * xi;
                } else if (xp != -1) {
                    result[xp + idx - (BASE_SIZE - 1)] += values[base + XP] * xi;
                }

                // -Y
                if ((mask & MIN_Y) == 0) {
                    result[row - BASE_SIZE * BASE_SIZE] += values[base + YM] * xi;
                } else if (ym != -1) {
                    result[ym + idx + Y_WRAP] += values[base + YM] * xi;
                }

                // +Y
                if ((mask & MAX_Y) == 0) {
                    result[row + BASE_SIZE * BASE_SIZE] += values[base + YP] * xi;
                } else if (yp != -1) {
                    result[yp + idx - Y_WRAP] += values[base + YP] * xi;
                }

                // -Z
                if ((mask & MIN_Z) == 0) {
                    result[row - BASE_SIZE] += values[base + ZM] * xi;
                } else if (zm != -1) {
                    result[zm + idx + Z_WRAP] += values[base + ZM] * xi;
                }

                // +Z
                if ((mask & MAX_Z) == 0) {
                    result[row + BASE_SIZE] += values[base + ZP] * xi;
                } else if (zp != -1) {
                    result[zp + idx - Z_WRAP] += values[base + ZP] * xi;
                }
            }
        }
    }

    @Override
    public int rows() {
        return blocks * BLOCK_SIZE;
    }

    @Override
    public int cols() {
        return blocks * BLOCK_SIZE;
    }

    @Override
    public double get(int row, int col) {
        //actually use c as direction, or use colIndex for that ? restore it so it can be use directly


        int block = row >> BASE_BITS * 3;
        int idx   = row & (BLOCK_SIZE - 1);


        int slot = findStencilSlot(block, idx, col);//this doesn't work at all

        return values[row * 7 + slot];

    }

    /*public int[] getRowCols(int r) {
        int base = r * NNZ_PER_ROW;

        int[] out = new int[NNZ_PER_ROW];
        System.arraycopy(colIndex, base, out, 0, NNZ_PER_ROW);

        return out;
    }*/

    public double[] getRowValues(int r) {
        int base = r * NNZ_PER_ROW;

        double[] out = new double[NNZ_PER_ROW];
        System.arraycopy(values, base, out, 0, NNZ_PER_ROW);

        return out;
    }
}