package com.rae.formicapi.init;

import com.mojang.brigadier.Command;
import com.rae.formicapi.FormicAPI;
import com.rae.formicapi.content.gui.screen.TestScreen;
import net.minecraft.client.Minecraft;
import net.minecraft.commands.Commands;
import net.minecraftforge.api.distmarker.Dist;
import net.minecraftforge.client.event.RegisterClientCommandsEvent;
import net.minecraftforge.event.RegisterCommandsEvent;
import net.minecraftforge.eventbus.api.SubscribeEvent;
import net.minecraftforge.fml.DistExecutor;
import net.minecraftforge.fml.common.Mod;

@Mod.EventBusSubscriber(modid = FormicAPI.MODID)
public class CommandsInit {
    @SubscribeEvent
    public static void registerCommands(RegisterCommandsEvent event) {
        // Root command: /formicapi
        /*dispatcher.register(
                Commands.literal("formicapi")
                        .then(Commands.literal("testScreen").executes(
                                context -> {
                                    ScreenOpener.open(new TestScreen());
                                    return Command.SINGLE_SUCCESS;
                                }
                        )
                )
        );*/
    }

    @SubscribeEvent
    public static void onRegisterClientCommands(RegisterClientCommandsEvent event) {
        event.getDispatcher().register(
                Commands.literal("formicapi").then(Commands.literal("testScreen").executes(
                        context -> {
                            DistExecutor.unsafeRunWhenOn(Dist.CLIENT, () -> () -> {
                                //Minecraft.getInstance().setScreen(new TestScreen());
                            });
                            return Command.SINGLE_SUCCESS;
                        }
                ))
        );
    }
}