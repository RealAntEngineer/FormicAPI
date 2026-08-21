package com.rae.formicapi.content.event;

import com.rae.formicapi.init.CommandsInit;
import com.rae.formicapi.content.data.managers.FloatMapDataLoader;
import com.rae.formicapi.content.thermal_utilities.FullTableBased;
import net.minecraft.client.Minecraft;
import net.minecraft.core.Registry;
import net.minecraft.core.RegistryAccess;
import net.minecraft.resources.ResourceKey;
import net.neoforged.bus.api.SubscribeEvent;
import net.neoforged.fml.common.EventBusSubscriber;
import net.neoforged.neoforge.event.RegisterCommandsEvent;
import net.neoforged.neoforge.event.entity.player.PlayerEvent;
import net.neoforged.neoforge.event.server.ServerStartedEvent;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.jetbrains.annotations.Nullable;

import java.util.Objects;

@EventBusSubscriber()
public class Event {
    private static final    Logger                LOGGER         = LogManager.getLogger();
    public static @Nullable RegistryAccess.Frozen registryAccess = null;

    @SubscribeEvent
    public static void onServerStarted(ServerStartedEvent event) {
        LOGGER.info("Getting the server registry access");
        registryAccess = event.getServer().registryAccess();
        FloatMapDataLoader.reloadRegistry();
    }

    @SubscribeEvent
    public static void onPlayerJoin(PlayerEvent.PlayerLoggedInEvent event){
        FullTableBased.onPlayerJoin(event);
    }


    public static <T> Registry<T> getSideAwareRegistry(ResourceKey<Registry<T>> registryKey) {
        if (registryAccess != null) {
            return registryAccess.registryOrThrow(registryKey);
        } else {
            LOGGER.debug("Getting the registry access from the client");
            var connection = Minecraft.getInstance().getConnection();
            if (connection == null) {
                LOGGER.error("Minecraft client connection unavailable - registry not yet synced");
                throw new IllegalStateException("Cannot access registry " + registryKey.location() +
                        " before client connection is established. This typically means you're trying to " +
                        "access a datapack registry too early during initialization.");
            }
            return connection.registryAccess()
                    .registry(registryKey)
                    .orElseThrow(() -> new IllegalStateException("Registry " + registryKey.location() + " not found"));
        }
    }

    @SubscribeEvent
    public static void onRegisterCommands(RegisterCommandsEvent event) {
        CommandsInit.register(event.getDispatcher());
    }
}