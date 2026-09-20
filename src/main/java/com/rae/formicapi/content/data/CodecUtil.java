package com.rae.formicapi.content.data;

import com.mojang.serialization.Codec;
import com.mojang.serialization.DataResult;
import com.mojang.serialization.codecs.RecordCodecBuilder;
import net.createmod.catnip.data.Pair;

import java.util.ArrayList;
import java.util.List;
import java.util.TreeMap;

public class CodecUtil {


    public static <K extends Comparable<K>, V> Codec<TreeMap<K, V>> orderedMapCodec(Codec<K> keyCodec, Codec<V> valueCodec) {
        return RecordCodecBuilder.<Pair<List<K>, List<V>>>create(instance ->
                instance.group(
                        keyCodec.listOf().fieldOf("keys").forGetter(Pair::getFirst),
                        valueCodec.listOf().fieldOf("values").forGetter(Pair::getSecond)
                ).apply(instance, Pair::of)
        ).flatXmap(
                pair -> {
                    List<K> keys = pair.getFirst();
                    List<V> values = pair.getSecond();
                    if (keys.size() != values.size()) {
                        return DataResult.error(() ->
                                "keys (" + keys.size() + ") and values (" + values.size() + ") size mismatch");
                    }
                    TreeMap<K, V> map = new TreeMap<>();
                    for (int i = 0; i < keys.size(); i++) {
                        map.put(keys.get(i), values.get(i));
                    }
                    return DataResult.success(map);
                },
                map -> DataResult.success(Pair.of(
                        new ArrayList<>(map.keySet()),
                        new ArrayList<>(map.values())
                ))
        );
    }
}
