package com.rae.formicapi.content.thermal_utilities;

import com.rae.formicapi.FormicAPI;
import com.rae.formicapi.content.data.managers.TwoDSparceTabulatedFunctionLoader;
import com.rae.formicapi.init.PacketInit;
import com.simibubi.create.foundation.networking.SimplePacketBase;
import net.minecraft.nbt.CompoundTag;
import net.minecraft.network.FriendlyByteBuf;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.entity.player.Player;
import net.minecraftforge.event.AddReloadListenerEvent;
import net.minecraftforge.event.entity.player.PlayerEvent;
import net.minecraftforge.network.NetworkEvent;
import net.minecraftforge.network.PacketDistributor;
import net.minecraftforge.network.simple.SimpleChannel;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.List;
import java.util.Objects;

public class FullTableBased {
    public static final SpecificRealGazState DEFAULT_STATE = new SpecificRealGazState(300f, 101_300f, 112_665f, 0f);

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
    public static SpecificRealGazState isentropicExpansion(SpecificRealGazState initial, float expansionFactor) {

        return isentropicPressureChange(initial.specificEnthalpy(), initial.pressure(), initial.pressure() / expansionFactor);
    }

    private static @NotNull SpecificRealGazState isentropicPressureChange(float H1, float P1, float finalPressure) {
        float sTarget = getS(H1, P1);

        // Initial guess from table interpolation
        float h = getH(sTarget, finalPressure);

        final float EPS = 1e-4f * P1;     // derivative step
        final float TOL = 1e-1f;     // convergence tolerance
        final int MAX_ITER = 20;

        for (int i = 0; i < MAX_ITER; i++) {
            float s = getS(h, finalPressure);
            float f = s - sTarget;

            if (Math.abs(f) < TOL) {
                break; // converged
            }

            // Numerical derivative ds/dh
            float s2 = getS(h + EPS, finalPressure);
            float df = (s2 - s) / EPS;

            // Safety check
            if (Math.abs(df) < 1e-8f) {
                break; // derivative too small → avoid explosion
            }

            h -= f / df;
        }

        float Tfinal = getT(h, finalPressure);
        float xFinal = getX(h, Tfinal);

        return new SpecificRealGazState(Tfinal, finalPressure, h, xFinal);
    }

    public static float getS(float H, float P) {
        if (!WATER_HP_S.loaded()) {
            FormicAPI.LOGGER.warn("Enthalpy | Pressure -> Entropy : table not loaded, returning a default value");
            return 1000;
        }
        return WATER_HP_S.getValue(H, P);
    }

    public static float getH(float S, float P) {
        if (!WATER_SP_H.loaded()) {
            FormicAPI.LOGGER.warn("Entropy | Pressure -> Enthalpy : table not loaded, returning a default value");
            return DEFAULT_STATE.specificEnthalpy();
        }
        return WATER_SP_H.getValue(S, P);
    }

    public static float getT(float H, float P) {
        if (!WATER_HP_T.loaded()) {
            FormicAPI.LOGGER.warn("Enthalpy | Pressure -> Temperature : table not loaded, returning a default value");
            return DEFAULT_STATE.temperature();
        }
        return WATER_HP_T.getValue(H, P);
    }

    public static float getX(float H, float P) {
        if (!WATER_HP_X.loaded()) {
            FormicAPI.LOGGER.warn("Enthalpy | Pressure -> Vapor quality : table not loaded, returning a default value");
            return DEFAULT_STATE.vaporQuality();
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
    public static SpecificRealGazState isentropicCompression(SpecificRealGazState initial, float compressionFactor) {

        return isentropicPressureChange(initial.specificEnthalpy(), initial.pressure(), initial.pressure() * compressionFactor);

    }

    public static SpecificRealGazState isobaricTransfer(SpecificRealGazState fluidState, float specific_heat) {
        if (specific_heat == 0) {
            return fluidState;
        } else {
            float newH = fluidState.specificEnthalpy() + specific_heat;
            float newPressure = fluidState.pressure();
            float newT = getT(newH, newPressure);
            float newVaporQuality = getX(newH, newPressure);
            return new SpecificRealGazState(newT, newPressure, newH, newVaporQuality);
        }
    }


    //TODO -> it seems to not be working when amount are too low -> protection against 0 values ?
    public static SpecificRealGazState mix(SpecificRealGazState first, float firstAmount, SpecificRealGazState second, float secondAmount) {
        //System.out.println("first "+first+ " second"+second);
        if (firstAmount == 0) return second;
        if (secondAmount == 0) return first;
        float P = first.pressure() * firstAmount / (firstAmount + secondAmount) + second.pressure() * secondAmount / (firstAmount + secondAmount);
        float h = first.specificEnthalpy() * firstAmount / (firstAmount + secondAmount) + second.specificEnthalpy() * secondAmount / (firstAmount + secondAmount);
        //float x = first.vaporQuality()*firstAmount/(firstAmount+ secondAmount) + second.vaporQuality()*secondAmount/(firstAmount+ secondAmount);
        SpecificRealGazState state = new SpecificRealGazState(
                getT(h, P), P, h, getX(h, P));
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

    public static void onPlayerJoin(PlayerEvent.PlayerLoggedInEvent event){
        Player player = event.getEntity();
        if (player instanceof ServerPlayer serverPlayer){
            //we need to cut this into smaller packets
            var channel = PacketInit.getChannel();

            // 1. Clear client tables first
            channel.send(
                    PacketDistributor.TRACKING_ENTITY_AND_SELF.with(() -> serverPlayer),
                    new ClearTablesPacket()
            );

            // 2. Send all tables in chunks
            sendTable(channel, serverPlayer, WATER_HP_T.splitSerialize(), TableType.HP_T);
            sendTable(channel, serverPlayer, WATER_HP_S.splitSerialize(), TableType.HP_S);
            sendTable(channel, serverPlayer, WATER_HP_X.splitSerialize(), TableType.HP_X);
            sendTable(channel, serverPlayer, WATER_SP_H.splitSerialize(), TableType.SP_H);
        }
    }

    private static void sendTable(SimpleChannel channel,
                                  ServerPlayer player,
                                  List<CompoundTag> chunks,
                                  TableType type) {

        for (CompoundTag tag : chunks) {
            channel.send(
                    PacketDistributor.TRACKING_ENTITY_AND_SELF.with(() -> player),
                    new SynchTablesPacket(tag, type)
            );
        }
    }

    public static class ClearTablesPacket extends SimplePacketBase {
        public ClearTablesPacket() {
        }

        public ClearTablesPacket(@NotNull FriendlyByteBuf buffer) {
        }

        public void write(FriendlyByteBuf buffer) {
        }

        public boolean handle(NetworkEvent.Context context) {
            context.enqueueWork(
                    () ->{
                        WATER_HP_T.clearFunction();
                        WATER_HP_S.clearFunction();
                        WATER_HP_X.clearFunction();
                        WATER_SP_H.clearFunction();
                    }
            );
            return true;
        }
    }

    public enum TableType {
        HP_T, HP_S, HP_X, SP_H
    }

    public static class SynchTablesPacket extends SimplePacketBase {
        private final @Nullable CompoundTag nbt;
        private final @Nullable TableType type;


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

        @Override
        public void write(FriendlyByteBuf buffer) {
            buffer.writeNbt(nbt);
            assert type != null;
            buffer.writeEnum(type);
        }

        @Override
        public boolean handle(NetworkEvent.Context context) {
            context.enqueueWork(
                    () -> {
                        switch (Objects.requireNonNull(type)) {
                            case HP_T -> WATER_HP_T.mergeFromNBT(nbt);
                            case HP_S -> WATER_HP_S.mergeFromNBT(nbt);
                            case HP_X -> WATER_HP_X.mergeFromNBT(nbt);
                            case SP_H -> WATER_SP_H.mergeFromNBT(nbt);
                        }
                    }
            );
            return true;
        }
    }
}