package com.rae.formicapi.content.event;


import com.rae.formicapi.FormicAPI;
import com.rae.formicapi.content.gui.screen.TestScreen;
import net.createmod.catnip.gui.ScreenOpener;
import net.minecraftforge.client.event.InputEvent;
import net.minecraftforge.client.event.ScreenEvent;
import net.minecraftforge.eventbus.api.SubscribeEvent;
import net.minecraftforge.fml.common.Mod;
import org.lwjgl.glfw.GLFW;

@Mod.EventBusSubscriber(modid = FormicAPI.MODID)
public class FormicEventHandler {

    @SubscribeEvent
    public static void keyPressed(InputEvent.Key event) {
        // Check if a specific key was pressed (e.g., 'L' key)
        int key = event.getKey();
        boolean pressed = !(event.getAction() == 0);

        if (GLFW.GLFW_KEY_F == key) {
            ScreenOpener.open(new TestScreen());
        }
    }
}
