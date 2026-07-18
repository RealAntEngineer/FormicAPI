/**
 * A lightweight multiblock framework built around a single controller block and a filler block.
 * <p>
 * Every multiblock has exactly one controller ("master") block implementing
 * {@link com.rae.formicapi.content.multiblock.IMBController} - either
 * {@link com.rae.formicapi.content.multiblock.MBController} for a plain directional block, or
 * {@link com.rae.formicapi.content.multiblock.MBKineticController} for one that participates in
 * Create's kinetic network. The controller defines the structure's footprint via
 * {@code getDefaultOffset}/{@code getDefaultSize}, which may vary per {@link
 * net.minecraft.world.level.block.state.BlockState} so the same controller can support multiple
 * structure shapes/sizes, and its overall collision shape via {@code getGlobalShape}.
 * <p>
 * The rest of the footprint is filled with {@link
 * com.rae.formicapi.content.multiblock.MBStructureBlock}s, each pointing back toward the
 * controller through a chain of {@code FACING} values
 * ({@link com.rae.formicapi.content.multiblock.MBStructureBlock#getMaster}). This filling happens
 * automatically when the controller is placed
 * ({@link com.rae.formicapi.content.multiblock.IMBController#repairStructure}), and structure
 * blocks forward their hit-box, interactions, and destruction back to the controller for as long
 * as that chain stays valid
 * ({@link com.rae.formicapi.content.multiblock.MBStructureBlock#stillValid}).
 * <p>
 * Placement itself goes through {@link com.rae.formicapi.content.multiblock.MBItem}, which
 * refuses to place the controller unless every position in its (state-dependent) bounding box is
 * clear, since placement fills that whole box with structure blocks.
 *
 * @see com.rae.formicapi.content.multiblock.IMBController
 * @see com.rae.formicapi.content.multiblock.MBStructureBlock
 */
@NonnullDefault
package com.rae.formicapi.content.multiblock;

import org.lwjgl.system.NonnullDefault;