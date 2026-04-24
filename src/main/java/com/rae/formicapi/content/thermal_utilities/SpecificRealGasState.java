package com.rae.formicapi.content.thermal_utilities;

import com.mojang.serialization.Codec;
import com.mojang.serialization.codecs.RecordCodecBuilder;
import com.rae.formicapi.FormicAPI;
import io.netty.buffer.ByteBuf;
import net.minecraft.nbt.CompoundTag;
import net.minecraft.network.FriendlyByteBuf;
import net.minecraft.network.codec.StreamCodec;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.Objects;

public class SpecificRealGasState {

    public static final SpecificRealGasState DEFAULT_STATE = new SpecificRealGasState(101_300f, 112_665f, null, null, null);
    public static final StreamCodec<ByteBuf, SpecificRealGasState> STREAM_CODEC = new StreamCodec<>() {
        public @NotNull SpecificRealGasState decode(@NotNull ByteBuf buffer) {
            return new SpecificRealGasState(Objects.requireNonNull(FriendlyByteBuf.readNbt(buffer)));
        }

        public void encode(@NotNull ByteBuf buffer, SpecificRealGasState state) {
            FriendlyByteBuf.writeNbt(buffer, state.serialize());
        }
    };
    public static final Codec<SpecificRealGasState>                CODEC        = RecordCodecBuilder.create(i ->
            i.group(
                            Codec.FLOAT.fieldOf("pressure").forGetter(p -> p.pressure),
                            Codec.FLOAT.fieldOf("specific_enthalpy").forGetter(p -> p.specificEnthalpy),
                            Codec.FLOAT.optionalFieldOf("temperature", null).forGetter(p -> p.temperature),
                            Codec.FLOAT.optionalFieldOf("specific_entropy", null).forGetter(p -> p.specificEntropy),
                            Codec.FLOAT.optionalFieldOf("vapor_quality", null).forGetter(p -> p.vaporQuality)
                    )
                    .apply(i, SpecificRealGasState::new));
    Float pressure;
    Float specificEnthalpy;
    Float temperature;
    Float specificEntropy;
    Float vaporQuality;

    public SpecificRealGasState(CompoundTag tag) {
        this(tag.getFloat("pressure"), tag.getFloat("specific_enthalpy"), tag.contains("temperature") ? tag.getFloat("temperature") : null,
                tag.contains("specific_entropy") ? tag.getFloat("specific_entropy") : null,
                tag.contains("vapor_quality") ? tag.getFloat("vapor_quality") : null);
    }

    public SpecificRealGasState(@NotNull Float pressure, @NotNull Float specificEnthalpy, @Nullable Float temperature,
                                @Nullable Float specificEntropy, @Nullable Float vaporQuality) {
        this.temperature = temperature == null ? null : Math.max(0, temperature);
        this.pressure = Math.max(0, pressure);
        this.specificEnthalpy = specificEnthalpy;
        this.specificEntropy = specificEntropy;
        if (vaporQuality != null && vaporQuality > 1) {
            FormicAPI.LOGGER.warn("vapor quality > 1 given, check your code");
        }
        this.vaporQuality = vaporQuality == null ? null : Math.clamp(vaporQuality, 0, 1);
    }

    @Override
    public @NotNull String toString() {
        return "SpecificRealGasState{" +
                "temperature=" + temperature +
                ", pressure=" + pressure +
                ", specific_enthalpy=" + specificEnthalpy +
                ", specific_entropy=" + specificEntropy +
                ", vaporQuality=" + vaporQuality +
                '}';
    }

    public CompoundTag serialize() {
        CompoundTag tag = new CompoundTag();
        if (temperature != null) tag.putFloat("temperature", temperature);
        if (pressure != null) tag.putFloat("pressure", pressure);
        if (specificEnthalpy != null) tag.putFloat("specific_enthalpy", specificEnthalpy);
        if (specificEntropy != null) tag.putFloat("specific_entropy", specificEntropy);
        if (vaporQuality != null) tag.putFloat("vapor_quality", vaporQuality);
        return tag;
    }

    public @NotNull Float temperature() {
        if (temperature == null) {
            temperature = FullTableBased.getT(pressure, specificEnthalpy);
        }
        return temperature;
    }

    public @NotNull Float specificEntropy() {
        if (specificEntropy == null) {
            specificEntropy = FullTableBased.getS(pressure, specificEnthalpy);
        }
        return specificEntropy;
    }

    public @NotNull Float vaporQuality() {
        if (vaporQuality == null) {
            vaporQuality = FullTableBased.getX(pressure, specificEnthalpy);
        }
        return vaporQuality;
    }

    public Float specificEnthalpy() {
        return specificEnthalpy;
    }

    public Float pressure() {
        return pressure;
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof SpecificRealGasState that)) return false;
        return Objects.equals(pressure, that.pressure) && Objects.equals(specificEnthalpy, that.specificEnthalpy);
    }

    @Override
    public int hashCode() {
        return Objects.hash(pressure, specificEnthalpy, temperature, specificEntropy, vaporQuality);
    }
}
