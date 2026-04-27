package com.rae.formicapi.content.thermal_utilities;

import com.rae.formicapi.FormicAPI;
import com.rae.formicapi.content.data.managers.TwoDSparceTabulatedFunctionLoader;
import com.rae.formicapi.init.PacketInit;
import net.createmod.catnip.net.base.ClientboundPacketPayload;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.nbt.CompoundTag;
import net.minecraft.network.FriendlyByteBuf;
import net.minecraft.network.RegistryFriendlyByteBuf;
import net.minecraft.network.codec.StreamCodec;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.entity.player.Player;
import net.neoforged.api.distmarker.Dist;
import net.neoforged.fml.loading.FMLEnvironment;
import net.neoforged.neoforge.event.AddReloadListenerEvent;
import net.neoforged.neoforge.event.entity.player.PlayerEvent;
import net.neoforged.neoforge.network.PacketDistributor;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.List;
import java.util.Objects;

import static com.rae.formicapi.content.thermal_utilities.SpecificRealGasState.DEFAULT_STATE;

public class FullTableBased {

    private static final TwoDSparceTabulatedFunctionLoader WATER_HP_T = new TwoDSparceTabulatedFunctionLoader(FormicAPI.MODID, "water/hp_to_t");
    private static final TwoDSparceTabulatedFunctionLoader WATER_HP_S = new TwoDSparceTabulatedFunctionLoader(FormicAPI.MODID, "water/hp_to_s");
    private static final TwoDSparceTabulatedFunctionLoader WATER_HP_X = new TwoDSparceTabulatedFunctionLoader(FormicAPI.MODID, "water/hp_to_x");
    private static final TwoDSparceTabulatedFunctionLoader WATER_SP_H = new TwoDSparceTabulatedFunctionLoader(FormicAPI.MODID, "water/sp_to_h");

    /**
     * adiabatic reversible expansion
     *
     * @param initial         : initial state
     * @param expansionFactor the initial pressure over the pressure of the fluid at the end of the turbine
     * @return the new fluid state
     */
    public static SpecificRealGasState isentropicExpansion(SpecificRealGasState initial, float expansionFactor) {

        return isentropicPressureChange(initial.pressure(), initial.specificEntropy(), initial.pressure() / expansionFactor);
    }

    private static @NotNull SpecificRealGasState isentropicPressureChange(float P1, float sTarget, float finalPressure) {

        // Initial guess from table interpolation
        float h = getH(finalPressure, sTarget);

        final float EPS      = 1e-4f * P1;     // derivative step
        final float TOL      = 1e-1f;     // convergence tolerance
        final int   MAX_ITER = 20;

        for (int i = 0; i < MAX_ITER; i++) {
            float s = getS(finalPressure, h);
            float f = s - sTarget;

            if (Math.abs(f) < TOL) {
                break; // converged
            }

            // Numerical derivative ds/dh
            float s2 = getS(finalPressure, h + EPS);
            float df = (s2 - s) / EPS;

            // Safety check
            if (Math.abs(df) < 1e-8f) {
                FormicAPI.LOGGER.info("early break in isentropic computation cause by derivative too small");
                break; // derivative too small → avoid explosion
            }

            h -= f / df;
        }

        float TFinal = getT(finalPressure, h);
        float xFinal = getX(finalPressure, h);
        float sFinal = getS(finalPressure, h);

        return new SpecificRealGasState(finalPressure, h, TFinal, sFinal, xFinal);
    }

    public static float getS(float P, float H) {
        if (!WATER_HP_S.loaded()) {
            FormicAPI.LOGGER.warn("Enthalpy | Pressure -> Entropy : table not loaded, returning a default value");
            return 350;
        }
        return WATER_HP_S.getValue(H, P);
    }

    public static float getH(float P, float S) {
        if (!WATER_SP_H.loaded()) {
            FormicAPI.LOGGER.warn("Entropy | Pressure -> Enthalpy : table not loaded, returning a default value");
            return 121_100;
        }
        return WATER_SP_H.getValue(S, P);
    }

    public static float getT(float P, float H) {
        if (!WATER_HP_T.loaded()) {
            FormicAPI.LOGGER.warn("Enthalpy | Pressure -> Temperature : table not loaded, returning a default value");
            return 300;
        }
        return WATER_HP_T.getValue(H, P);
    }

    public static float getX(float P, float H) {
        if (!WATER_HP_X.loaded()) {
            FormicAPI.LOGGER.warn("Enthalpy | Pressure -> Vapor quality : table not loaded, returning a default value");
            return 0;
        }
        return WATER_HP_X.getValue(H, P);
    }

    /**
     * adiabatic reversible compression
     *
     * @param initial           : initial state
     * @param compressionFactor :the pressure of the fluid at the end of the turbine over the initial pressure
     * @return the new fluid state
     */
    public static SpecificRealGasState isentropicCompression(SpecificRealGasState initial, float compressionFactor) {

        return isentropicPressureChange(initial.pressure(), initial.specificEntropy(), initial.pressure() * compressionFactor);

    }

    public static SpecificRealGasState isobaricTransfer(SpecificRealGasState fluidState, float specific_heat) {
        if (specific_heat == 0) {
            return fluidState;
        } else {
            float newH            = fluidState.specificEnthalpy() + specific_heat;
            float newPressure     = fluidState.pressure();
            float newT            = getT(newPressure, newH);
            float newVaporQuality = getX(newPressure, newH);
            return new SpecificRealGasState(newPressure, newH, newT, getS(newPressure, newH), newVaporQuality);
        }
    }


    //TODO -> it seems to not be working when amount are too low -> protection against 0 values ?
    public static SpecificRealGasState mix(SpecificRealGasState first, float firstAmount, SpecificRealGasState second, float secondAmount) {
        //System.out.println("first "+first+ " second"+second);
        if (firstAmount == 0) return second;
        if (secondAmount == 0) return first;
        float P = first.pressure() * firstAmount / (firstAmount + secondAmount) + second.pressure() * secondAmount / (firstAmount + secondAmount);
        float h = first.specificEnthalpy() * firstAmount / (firstAmount + secondAmount) + second.specificEnthalpy() * secondAmount / (firstAmount + secondAmount);
        //float x = first.vaporQuality()*firstAmount/(firstAmount+ secondAmount) + second.vaporQuality()*secondAmount/(firstAmount+ secondAmount);
        SpecificRealGasState state = new SpecificRealGasState(
                P, h, getT(P, h), getS(P, h), getX(P, h));
        //System.out.println(state);
        if (first.temperature().isNaN() || second.temperature().isNaN()) {
            return DEFAULT_STATE;
        }
        if (first.temperature() > 20_000 || second.temperature() > 20_000) {
            return DEFAULT_STATE;
        }
        return state;
    }

    public static void addReloadListeners(AddReloadListenerEvent event) {
        event.addListener(WATER_HP_T);
        event.addListener(WATER_HP_S);
        event.addListener(WATER_HP_X);
        event.addListener(WATER_SP_H);
    }

    public static void onPlayerJoin(PlayerEvent.PlayerLoggedInEvent event) {
        //ONLY DO THIS IF IT'S A DISTANT SERVER
        if (FMLEnvironment.dist == Dist.DEDICATED_SERVER) {
            Player player = event.getEntity();
            if (player instanceof ServerPlayer serverPlayer) {

                List<CompoundTag> chunkHPT = WATER_HP_T.splitSerialize();
                List<CompoundTag> chunkHPS = WATER_HP_S.splitSerialize();
                List<CompoundTag> chunkHPX = WATER_HP_X.splitSerialize();
                List<CompoundTag> chunkSPH = WATER_SP_H.splitSerialize();

                FormicAPI.LOGGER.info("Asking Player to clear it's tables");
                PacketDistributor.sendToPlayer(serverPlayer,
                        new ClearTablesPacket());

                FormicAPI.LOGGER.info("Sending tables to the client");
                sendTable(serverPlayer, chunkHPT, TableType.HP_T);
                sendTable(serverPlayer, chunkHPS, TableType.HP_S);
                sendTable(serverPlayer, chunkHPX, TableType.HP_X);
                sendTable(serverPlayer, chunkSPH, TableType.SP_H);
            }
        }
    }

    private static void sendTable(ServerPlayer player,
                                  List<CompoundTag> chunks,
                                  TableType type) {

        for (CompoundTag tag : chunks) {
            PacketDistributor.sendToPlayer(player,
                    new SynchTablesPacket(tag, type)
            );
        }
    }

    public enum TableType {
        HP_T, HP_S, HP_X, SP_H
    }

    public static class ClearTablesPacket implements ClientboundPacketPayload {

        public static final StreamCodec<RegistryFriendlyByteBuf, ClearTablesPacket> STREAM_CODEC = StreamCodec.of(
                (byteBuf, packet) -> packet.write(byteBuf),
                ClearTablesPacket::new);

        public ClearTablesPacket() {
        }

        public ClearTablesPacket(@NotNull FriendlyByteBuf buffer) {
        }

        public void write(FriendlyByteBuf buffer) {
        }

        public void handle(LocalPlayer player) {

            WATER_HP_T.clearFunction();
            WATER_HP_S.clearFunction();
            WATER_HP_X.clearFunction();
            WATER_SP_H.clearFunction();

        }

        @Override
        public PacketTypeProvider getTypeProvider() {
            return PacketInit.CLEAR_TABLES;
        }
    }

    public static class SynchTablesPacket implements ClientboundPacketPayload {
        public static final     StreamCodec<RegistryFriendlyByteBuf, SynchTablesPacket> STREAM_CODEC = StreamCodec.of(
                (byteBuf, packet) -> packet.write(byteBuf),
                SynchTablesPacket::new);
        private final @Nullable CompoundTag                                             nbt;
        private final @Nullable TableType                                               type;

        // Construct from server data
        public SynchTablesPacket(@NotNull CompoundTag nbt, @NotNull TableType type) {
            this.nbt = nbt;
            this.type = type;
        }

        // Construct from network buffer
        public SynchTablesPacket(@NotNull FriendlyByteBuf buffer) {
            this.nbt = buffer.readNbt();
            this.type = buffer.readEnum(TableType.class);
        }

        public void write(RegistryFriendlyByteBuf buffer) {
            buffer.writeNbt(nbt);
            assert type != null;
            buffer.writeEnum(type);
        }

        @Override
        public void handle(LocalPlayer player) {

            switch (Objects.requireNonNull(type)) {
                case HP_T -> WATER_HP_T.mergeFromNBT(nbt);
                case HP_S -> WATER_HP_S.mergeFromNBT(nbt);
                case HP_X -> WATER_HP_X.mergeFromNBT(nbt);
                case SP_H -> WATER_SP_H.mergeFromNBT(nbt);
            }


        }

        @Override
        public PacketTypeProvider getTypeProvider() {
            return PacketInit.SYNCH_TABLES;
        }
    }
}