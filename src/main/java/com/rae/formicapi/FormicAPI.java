package com.rae.formicapi;

import com.rae.formicapi.content.config.FormicAPIConfigs;
import com.rae.formicapi.content.thermal_utilities.FullTableBased;
import com.rae.formicapi.init.PacketInit;
import net.minecraft.resources.ResourceLocation;
import net.minecraftforge.common.MinecraftForge;
import net.minecraftforge.event.AddReloadListenerEvent;
import net.minecraftforge.eventbus.api.IEventBus;
import net.minecraftforge.fml.ModLoadingContext;
import net.minecraftforge.fml.common.Mod;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

@Mod(FormicAPI.MODID)
public class FormicAPI {
    public static final String MODID = "formicapi";
    public static final Logger LOGGER = LogManager.getLogger(MODID);

    public FormicAPI() {
        IEventBus forgeEventBus = MinecraftForge.EVENT_BUS;
        ModLoadingContext modLoadingContext = ModLoadingContext.get();

        FormicAPIConfigs.registerConfigs(modLoadingContext);
        PacketInit.registerPackets();

        forgeEventBus.addListener(FormicAPI::onAddReloadListeners);
    }

    public static void onAddReloadListeners(AddReloadListenerEvent event) {
        FullTableBased.addReloadListeners(event);
    }

    public static ResourceLocation resource(String name) {
        return new ResourceLocation(MODID, name);
    }

}
