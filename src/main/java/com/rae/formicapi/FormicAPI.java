package com.rae.formicapi;

import com.rae.formicapi.config.FormicAPIConfigs;
import net.minecraftforge.common.MinecraftForge;
import net.minecraftforge.eventbus.api.IEventBus;
import net.minecraftforge.fml.ModLoadingContext;
import net.minecraftforge.fml.common.Mod;
import net.minecraftforge.fml.javafmlmod.FMLJavaModLoadingContext;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

@Mod(FormicAPI.MODID)
public class FormicAPI {
    public static final String MODID = "formicapi";
    public static final Logger LOGGER = LogManager.getLogger(MODID);
    public FormicAPI() {
        IEventBus modEventBus = FMLJavaModLoadingContext.get().getModEventBus();
        IEventBus forgeEventBus = MinecraftForge.EVENT_BUS;
        ModLoadingContext modLoadingContext = ModLoadingContext.get();
        FormicAPIConfigs.registerConfigs(modLoadingContext);

    }
}
