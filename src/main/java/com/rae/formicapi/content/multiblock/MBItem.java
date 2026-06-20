package com.rae.formicapi.content.multiblock;

import net.minecraft.core.BlockPos;
import net.minecraft.world.item.BlockItem;
import net.minecraft.world.item.context.BlockPlaceContext;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.state.BlockState;
import org.lwjgl.system.NonnullDefault;

/**
 * {@link BlockItem} for an {@link IMBController} block. Before placing the controller, this
 * checks that the controller's entire (state-dependent) bounding box - not just the clicked
 * position - is clear of obstructions, since placing the controller also fills the rest of the
 * box with {@link MBStructureBlock}s.
 */
@NonnullDefault
@SuppressWarnings("unused")
public class MBItem extends BlockItem {
    /** @throws IllegalArgumentException if {@code block} doesn't implement {@link IMBController} */
    public MBItem(Block block, Properties properties) {
        super(block, properties);
        if (!(block instanceof IMBController)) {
            throw new IllegalArgumentException("block must be an instance of IMBController for a MBItem");
        }
    }

    /**
     * Checks every position inside the controller's bounding box (per {@code pState}, the state
     * that will actually be placed) is air before allowing placement.
     */
    @Override
    protected boolean canPlace(BlockPlaceContext placingContext, BlockState placedState) {
        IMBController main    = (IMBController) getBlock();
        Level         lvl     = placingContext.getLevel();
        BlockPos      mainPos = placingContext.getClickedPos();
        return main.hasSpace(lvl, placedState, mainPos, false);
    }

    /**
     * Places the controller via {@code setBlockAndUpdate} (sends neighbor + client updates).
     * {@code setPlacedBy} and the placement advancement trigger are deliberately <i>not</i> called
     * here - {@code BlockItem.place} already calls both, with its own "is this still the block we
     * placed" check, immediately after this method returns. Duplicating that call here would fire
     * {@code setPlacedBy} - and therefore {@code repairStructure} - twice per placement.
     */
    @Override
    protected boolean placeBlock(BlockPlaceContext pContext, BlockState pState) {
        return pContext.getLevel().setBlockAndUpdate(pContext.getClickedPos(), pState);
    }
}