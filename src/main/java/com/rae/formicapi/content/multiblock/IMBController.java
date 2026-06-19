package com.rae.formicapi.content.multiblock;

import com.rae.formicapi.FormicAPI;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.core.Vec3i;
import net.minecraft.world.level.BlockGetter;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.block.DirectionalBlock;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.phys.shapes.CollisionContext;
import net.minecraft.world.phys.shapes.VoxelShape;

import java.util.ArrayDeque;
import java.util.HashSet;
import java.util.Queue;
import java.util.Set;

/**
 * Marks a block as the controller ("master") of a multiblock structure built out of
 * {@link MBStructureBlock} blocks. The controller owns the structure's shape: every
 * {@link MBStructureBlock} in the structure points back at this block and forwards
 * shape/interaction queries to it via {@link MBStructureBlock#getMaster}.
 * <p>
 * The structure's footprint is described relative to the controller by
 * {@link #getDefaultOffset(BlockState)} (where the controller sits inside the structure's
 * bounding box, assuming it faces {@link Direction#NORTH}) and {@link #getDefaultSize(BlockState)}
 * (the bounding box dimensions). Both are state-aware: a child block class can override them to
 * return different offsets/sizes for different {@link BlockState}s (e.g. a size or tier property),
 * letting the same controller block produce differently-shaped structures. {@link #getOffset} and
 * {@link #getSize} then rotate those defaults to match the controller's actual facing.
 */
public interface IMBController {
    /**
     * Returns this controller's full multiblock shape, expressed in world space relative to
     * {@code pos} (the controller's own position). Implementations typically build this from
     * {@link #getOffset} and {@link #getSize} for the controller's facing and state, so the
     * result can vary per-{@link BlockState} (e.g. different sizes/tiers). Callers (this
     * controller's own {@code getShape}, and {@link MBStructureBlock#getShape}) intersect the
     * returned shape with a single block's space before using it.
     */
    VoxelShape getGlobalShape(BlockState state, BlockGetter worldIn, BlockPos pos, CollisionContext context);

    /**
     * Walks outward from the controller (breadth-first) and fills every position inside its
     * state-dependent bounding box that isn't already a structure block with one, each pointing
     * back toward the controller via its {@code FACING} property. Used after the controller is
     * placed to (re)build the rest of the multiblock. No-ops on the client and bails out with a
     * warning after visiting 100 positions, as a safety net against malformed/oversized shapes.
     */
    default void repairStructure(Level level, BlockState state, BlockPos controlPos, Direction facing) {
        if (level.isClientSide()) return;
        MBStructureBlock structure = getStructure();
        Set<BlockPos>    visited   = new HashSet<>();
        Queue<Node>      toVisit   = new ArrayDeque<>();

        Vec3i    off       = getOffset(state, facing, false);
        Vec3i    size      = getSize(state, facing);
        BlockPos minCorner = controlPos.offset(off);

        for (Direction dir : Direction.values()) {
            BlockPos neighborPos = controlPos.relative(dir);

            if (isInsideBounds(neighborPos, minCorner, size)) {
                toVisit.add(new Node(dir, neighborPos));
            }
        }
        int i = 0;
        while (!toVisit.isEmpty()) {
            Node       node    = toVisit.poll();
            BlockState current = level.getBlockState(node.pos);


            if (!current.is(structure)) {
                level.setBlockAndUpdate(node.pos, structure.defaultBlockState().setValue(DirectionalBlock.FACING, node.fromDir.getOpposite()));
                visited.add(node.pos);

                for (Direction dir : Direction.values()) {
                    BlockPos neighborPos = node.pos.relative(dir);
                    //System.out.println(neighborPos);

                    if (isInsideBounds(neighborPos, minCorner, size) && !neighborPos.equals(controlPos) && !visited.contains(neighborPos)) {
                        toVisit.add(new Node(dir, neighborPos));
                    }
                }
            }
            i++;
            if (i > 100) {
                FormicAPI.LOGGER.warn("More than 100 blocks");
                break;
            }
        }
    }

    /** The {@link MBStructureBlock} variant used to fill out the rest of this multiblock. */
    MBStructureBlock getStructure();

    /**
     * Rotates {@link #getDefaultOffset(BlockState)} to account for the controller's facing,
     * yielding the controller's actual position within its bounding box. If {@code mirrorOnDir}
     * is {@code true}, the offset's east-west component is mirrored for blocks facing the
     * negative axis direction (useful for shapes that aren't symmetric on that axis).
     */
    default Vec3i getOffset(BlockState state, Direction facing, boolean mirrorOnDir) {
        final int dirMultiply   = facing.getAxisDirection() == Direction.AxisDirection.NEGATIVE && mirrorOnDir ? -1 : 1;
        Vec3i     defaultOffset = getDefaultOffset(state);
        return switch (facing.getAxis()) {
            case Z -> new Vec3i(defaultOffset.getZ(), defaultOffset.getY(), dirMultiply * defaultOffset.getX());
            case Y -> new Vec3i(defaultOffset.getY(), dirMultiply * defaultOffset.getX(), defaultOffset.getZ());
            default -> new Vec3i(dirMultiply * defaultOffset.getX(), defaultOffset.getY(), defaultOffset.getZ());
        };
    }

    /**
     * Rotates {@link #getDefaultSize(BlockState)} to account for the controller's facing,
     * yielding the actual width/height/depth of the structure's bounding box in world space.
     */
    default Vec3i getSize(BlockState state, Direction facing) {
        Vec3i defaultSize = getDefaultSize(state);
        return switch (facing.getAxis()) {
            case Z -> new Vec3i(defaultSize.getZ(), defaultSize.getY(), defaultSize.getX());
            case Y -> new Vec3i(defaultSize.getY(), defaultSize.getX(), defaultSize.getZ());
            default -> defaultSize;
        };
    }

    /** Whether {@code pos} falls within the axis-aligned box defined by {@code minCorner} and {@code size}. */
    default boolean isInsideBounds(BlockPos pos, BlockPos minCorner, Vec3i size) {
        int dx = minCorner.getX() - pos.getX();
        int dy = minCorner.getY() - pos.getY();
        int dz = minCorner.getZ() - pos.getZ();
        return dx >= 0 && dx < size.getX() &&
                dy >= 0 && dy < size.getY() &&
                dz >= 0 && dz < size.getZ();
    }

    /**
     * The controller's position within the structure's bounding box, assuming the controller
     * faces {@link Direction#NORTH}. Override this per-{@code state} (e.g. switching on a custom
     * size/tier {@link BlockState} property) to give different states different shapes.
     */
    Vec3i getDefaultOffset(BlockState state);

    /**
     * The structure's bounding box dimensions, assuming the controller faces
     * {@link Direction#NORTH}. Override this per-{@code state} alongside
     * {@link #getDefaultOffset(BlockState)} to give different states different shapes.
     */
    Vec3i getDefaultSize(BlockState state);

    /** A position still to be visited by {@link #repairStructure}, and the direction it was reached from. */
    record Node(Direction fromDir, BlockPos pos) {
    }
}