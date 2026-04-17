package com.rae.formicapi.init;

import com.rae.formicapi.FormicAPI;
import com.rae.formicapi.content.thermal_utilities.FullTableBased;
import net.createmod.catnip.net.base.BasePacketPayload;
import net.createmod.catnip.net.base.CatnipPacketRegistry;
import net.minecraft.network.RegistryFriendlyByteBuf;
import net.minecraft.network.codec.StreamCodec;
import net.minecraft.network.protocol.common.custom.CustomPacketPayload;

import java.util.Locale;

public enum PacketInit implements BasePacketPayload.PacketTypeProvider{
    SYNCH_TABLES(FullTableBased.SynchTablesPacket.class, FullTableBased.SynchTablesPacket.STREAM_CODEC),
    CLEAR_TABLES(FullTableBased.ClearTablesPacket.class, FullTableBased.ClearTablesPacket.STREAM_CODEC);

    private final CatnipPacketRegistry.PacketType<?> type;

    <T extends BasePacketPayload> PacketInit(Class<T> clazz, StreamCodec<? super RegistryFriendlyByteBuf, T> codec) {
        String name = this.name().toLowerCase(Locale.ROOT);
        this.type = new CatnipPacketRegistry.PacketType<>(
                new CustomPacketPayload.Type<>(FormicAPI.resource(name)),
                clazz, codec
        );
    }

    @Override
    @SuppressWarnings("unchecked")
    public <T extends CustomPacketPayload> CustomPacketPayload.Type<T> getType() {
        return (CustomPacketPayload.Type<T>) this.type.type();
    }

    public static void register() {
        CatnipPacketRegistry packetRegistry = new CatnipPacketRegistry(FormicAPI.MODID, 1);
        for (PacketInit packet : PacketInit.values()) {
            packetRegistry.registerPacket(packet.type);
        }
        packetRegistry.registerAllPackets();
    }

}
