package com.rae.formicapi;

import com.rae.formicapi.config.FormicAPIConfigs;
import com.rae.formicapi.data.managers.TwoDTabulatedFunctionLoader;
import com.rae.formicapi.data.providers.TwoDTabulatedFunctionProvider;
import com.rae.formicapi.math.data.StepMode;
import com.rae.formicapi.new_thermalmodels.FullTableBased;
import com.rae.formicapi.thermal_utilities.helper.WaterCubicEOS;
import net.minecraft.data.DataGenerator;
import net.minecraft.data.PackOutput;
import net.minecraft.resources.ResourceLocation;
import net.minecraftforge.common.MinecraftForge;
import net.minecraftforge.data.event.GatherDataEvent;
import net.minecraftforge.event.AddReloadListenerEvent;
import net.minecraftforge.event.RegisterCommandsEvent;
import net.minecraftforge.eventbus.api.EventPriority;
import net.minecraftforge.eventbus.api.IEventBus;
import net.minecraftforge.eventbus.api.SubscribeEvent;
import net.minecraftforge.fml.ModLoadingContext;
import net.minecraftforge.fml.common.Mod;
import net.minecraftforge.fml.javafmlmod.FMLJavaModLoadingContext;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

@Mod(FormicAPI.MODID)
public class FormicAPI {
    public static final String MODID = "formicapi";
    public static final Logger LOGGER = LogManager.getLogger(MODID);
    public static final TwoDTabulatedFunctionLoader WATER_PH_T = new TwoDTabulatedFunctionLoader(MODID,"water/pressure_enthalpy_to_temperature");
    public static final TwoDTabulatedFunctionLoader WATER_PS_T = new TwoDTabulatedFunctionLoader(MODID,"water/pressure_entropy_to_temperature");

    public FormicAPI() {
        IEventBus modEventBus = FMLJavaModLoadingContext.get().getModEventBus();
        IEventBus forgeEventBus = MinecraftForge.EVENT_BUS;
        ModLoadingContext modLoadingContext = ModLoadingContext.get();
        FormicAPIConfigs.registerConfigs(modLoadingContext);
        modEventBus.addListener(EventPriority.NORMAL,this::gatherData);
        forgeEventBus.addListener(FormicAPI::onAddReloadListeners);


    }

    public static void onAddReloadListeners(AddReloadListenerEvent event)
    {
        event.addListener(FormicAPI.WATER_PH_T);
        event.addListener(FormicAPI.WATER_PS_T);
        FullTableBased.addReloadListeners(event);
    }

    public void gatherData(GatherDataEvent event) {
        DataGenerator generator = event.getGenerator();
        PackOutput output = generator.getPackOutput();

        /*generator.addProvider(event.includeServer(), new TwoDTabulatedFunctionProvider(
                output, resource("water/pressure_enthalpy_to_temperature"),
                WaterCubicEOS::get_T
                ,1e2f,0, 3e7f, 2e7f,300,1000,StepMode.LOGARITHMIC, StepMode.LINEAR, true
        ));*/

        /*generator.addProvider(
                true, new TwoDTabulatedFunctionProvider(
                        output, resource("water/pressure_enthalpy_to_entropy"),
                        (P, H) -> {
                            //get H from the
                            float T = WaterCubicEOS.get_T(P,H);
                            CubicEOS EOS = EOSLibrary.getPRWaterEOS();
                            return (float) EOS.getEntropy(T, P, WaterCubicEOS.get_x(H,T,P));
                        }
                        ,1e2f,0, 3e7f, 2e7f,30,100,StepMode.LOGARITHMIC, StepMode.LINEAR, true

                )
        );*/
        /*generator.addProvider(
                event.includeServer(), new TwoDTabulatedFunctionProvider(
                        output, resource("water/pressure_entropy_to_temperature"),
                        (P, S) -> {
                            //get H from the
                            float initialH = 0;
                            float initialT = WaterCubicEOS.get_T(P, initialH);
                            float x = WaterCubicEOS.get_x(P, initialH, initialT);
                            return WaterCubicEOS.getT(initialT, x, P, S);
                        }
                        ,1e2f,1e-3f, 2e7f, 10e3f,30,100,StepMode.LOGARITHMIC, StepMode.LINEAR, false
                )
        );*/
        /*generator.addProvider(event.includeServer(), new TwoDTabulatedFunctionProvider(
                output, new ResourceLocation(MODID, "water/pressure_entropy_to_temperature"),
                (h, p)-> WaterCubicEOS.get_T(p, h)
                ,1e2f,0, 1e7f, 5e7f,1000,1000,StepMode.LOGARITHMIC, StepMode.LINEAR, true
        ));*/
    }
    public static ResourceLocation resource(String name) {
        return new ResourceLocation(MODID,name);
    }

}
