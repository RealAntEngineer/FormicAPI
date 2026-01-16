package com.rae.formicapi;

import com.mojang.brigadier.CommandDispatcher;
import com.mojang.brigadier.arguments.FloatArgumentType;
import com.mojang.brigadier.context.CommandContext;
import com.rae.formicapi.thermal_utilities.Plotting;
import net.minecraft.Util;
import net.minecraft.commands.CommandSourceStack;
import net.minecraft.commands.Commands;
import net.minecraft.network.chat.Component;

import java.util.stream.DoubleStream;

public class CommandsInit {

    public static void register(CommandDispatcher<CommandSourceStack> dispatcher) {
        dispatcher.register(Commands.literal("plotIsentropes")
                .then(Commands.argument("sMin", FloatArgumentType.floatArg(0))
                        .then(Commands.argument("sMax", FloatArgumentType.floatArg(0))
                                .then(Commands.argument("sStep", FloatArgumentType.floatArg(0))
                                        .then(Commands.argument("pMin", FloatArgumentType.floatArg(0))
                                                .then(Commands.argument("pMax", FloatArgumentType.floatArg(0))
                                                        .executes(CommandsInit::execute)
                                                )
                                        )
                                )
                        )
                )
        );
    }

    private static int execute(CommandContext<CommandSourceStack> ctx) {
        float sMin = FloatArgumentType.getFloat(ctx, "sMin");
        float sMax = FloatArgumentType.getFloat(ctx, "sMax");
        float sStep = FloatArgumentType.getFloat(ctx, "sStep");

        float pMin = FloatArgumentType.getFloat(ctx, "pMin");
        float pMax = FloatArgumentType.getFloat(ctx, "pMax");

        // Parse comma-separated entropy values, e.g., "3.5,4.0,4.5"
        double[] entropies = DoubleStream.iterate(
                sMin,
                s -> s <= sMax + 1e-6,
                s -> s + sStep
        ).toArray();
        //DistExecutor.unsafeRunWhenOn(
        //        Dist.CLIENT,() -> () ->

        Util.backgroundExecutor().execute(() -> {
            //Plotting.plotSaturationPressurePT("Water Saturation Pressure (P–H)", pMin, pMax);
            //Plotting.plotSaturationPressurePH("Water Saturation Pressure (P–H)", pMin, pMax);

            Plotting.plotIsentropesPH(
                    "Water Isentropes (P–H)",
                    entropies,
                    pMin,
                    pMax
            );


        });

        ctx.getSource().sendSuccess( () -> Component.literal("Isentrope plot generated!"),
                true
        );

        return 1;
    }


}
