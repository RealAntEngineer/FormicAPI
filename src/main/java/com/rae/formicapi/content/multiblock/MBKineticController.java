package com.rae.formicapi.content.multiblock;

import com.simibubi.create.content.kinetics.base.DirectionalKineticBlock;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.context.BlockPlaceContext;
import net.minecraft.world.level.BlockGetter;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.phys.shapes.BooleanOp;
import net.minecraft.world.phys.shapes.CollisionContext;
import net.minecraft.world.phys.shapes.Shapes;
import net.minecraft.world.phys.shapes.VoxelShape;
import org.jetbrains.annotations.Nullable;
import org.lwjgl.system.NonnullDefault;

import java.util.Objects;

/**
 * The main (controller/master) block for a kinetic (Create-powered) multiblock - identical in
 * role to {@link MBController}, but built on Create's {@link DirectionalKineticBlock} so it can
 * participate in Create's rotational power network. This is the block used for the model; to
 * make the model line up you'll need to look at the default offset/size given by your subclass
 * for the size you've chosen.
 * <p>
 * Subclasses are expected to implement {@link IMBController#getGlobalShape},
 * {@link IMBController#getDefaultOffset(BlockState)} and
 * {@link IMBController#getDefaultSize(BlockState)} - the latter two may vary their result based
 * on the given {@link BlockState} if this controller supports multiple structure shapes/sizes.
 */
@NonnullDefault
@SuppressWarnings("unused")
public abstract class MBKineticController extends DirectionalKineticBlock implements IMBController {
    final MBStructureBlock structure;

    /** @param structure the {@link MBStructureBlock} used to fill out the rest of this multiblock */
    protected MBKineticController(Properties properties, MBStructureBlock structure) {
        super(properties);
        this.structure = structure;
        this.registerDefaultState(this.defaultBlockState().setValue(FACING, Direction.NORTH));
    }

    /** Fills out the rest of the multiblock with {@link MBStructureBlock}s once the controller is placed. */
    @Override
    public void setPlacedBy(Level worldIn, BlockPos pos, BlockState state, @Nullable LivingEntity entity, ItemStack stack) {
        super.setPlacedBy(worldIn, pos, state, entity, stack);
        repairStructure(worldIn, state, pos, state.getValue(FACING));
    }

    /** Faces the placed block toward whichever side the player clicked. */
    public BlockState getStateForPlacement(BlockPlaceContext context) {
        return Objects.requireNonNull(super.getStateForPlacement(context)).setValue(FACING, context.getClickedFace());

    }

    /** Clips {@link IMBController#getGlobalShape} down to this single block's space. */
    @Override
    public VoxelShape getShape(BlockState state, BlockGetter level, BlockPos pos, CollisionContext context) {
        return Shapes.join(getGlobalShape(state, level, pos, context), Shapes.block(), BooleanOp.AND);
    }

    public MBStructureBlock getStructure() {
        return structure;
    }
}