package com.rae.formicapi;

import com.rae.formicapi.config.FormicAPIConfigs;
import com.rae.formicapi.thermal_utilities.FullTableBased;
import net.minecraft.resources.ResourceLocation;
import net.neoforged.bus.api.IEventBus;
import net.neoforged.fml.ModContainer;
import net.neoforged.fml.ModLoadingContext;
import net.neoforged.fml.common.Mod;
import net.neoforged.neoforge.common.NeoForge;
import net.neoforged.neoforge.event.AddReloadListenerEvent;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

@Mod(FormicAPI.MODID)
public class FormicAPI {
    public static final String MODID = "formicapi";
    public static final Logger LOGGER = LogManager.getLogger(MODID);

    public FormicAPI(IEventBus modEventBus, ModContainer modContainer) {
        IEventBus forgeEventBus = NeoForge.EVENT_BUS;
        ModLoadingContext modLoadingContext = ModLoadingContext.get();

        forgeEventBus.addListener(FormicAPI::onAddReloadListeners);
        FormicAPIConfigs.register(modLoadingContext, modContainer);



    }

    public static void onAddReloadListeners(AddReloadListenerEvent event)
    {
        FullTableBased.addReloadListeners(event);
    }


    public static ResourceLocation resource(String name) {
        return ResourceLocation.fromNamespaceAndPath(MODID,name);
    }

}
