package com.rae.formicapi;

import com.mojang.brigadier.CommandDispatcher;
import com.mojang.brigadier.arguments.FloatArgumentType;
import com.mojang.brigadier.builder.LiteralArgumentBuilder;
import com.mojang.brigadier.context.CommandContext;
import com.rae.formicapi.content.thermal_utilities.FullTableBased;
import com.rae.formicapi.content.thermal_utilities.Plotting;
import net.minecraft.Util;
import net.minecraft.commands.CommandSourceStack;
import net.minecraft.commands.Commands;
import net.minecraft.network.chat.Component;

import java.util.stream.DoubleStream;

public class CommandsInit {

    public static void register(CommandDispatcher<CommandSourceStack> dispatcher) {
        dispatcher.register(
                Commands.literal("formicapi")
                        .then(isotropesCommand())
                        .then(getSCommand())
                        .then(getHCommand())
                        .then(getTCommand())
                        .then(getXCommand())
        );

    }

    private static LiteralArgumentBuilder<CommandSourceStack> isotropesCommand() {
        return Commands.literal("plotIsentropes")
                .then(Commands.argument("sMin", FloatArgumentType.floatArg(0))
                        .then(Commands.argument("sMax", FloatArgumentType.floatArg(0))
                                .then(Commands.argument("sStep", FloatArgumentType.floatArg(0))
                                        .then(Commands.argument("pMin", FloatArgumentType.floatArg(0))
                                                .then(Commands.argument("pMax", FloatArgumentType.floatArg(0))
                                                        .executes(CommandsInit::executeIsotropesCommand)
                                                )
                                        )
                                )
                        )
                );
    }

    private static LiteralArgumentBuilder<CommandSourceStack> getSCommand() {
        return Commands.literal("getS")
                .then(Commands.argument("P", FloatArgumentType.floatArg(0))
                        .then(Commands.argument("H", FloatArgumentType.floatArg(0))
                                .executes(ctx -> {
                                    float P = FloatArgumentType.getFloat(ctx, "P");
                                    float H = FloatArgumentType.getFloat(ctx, "H");

                                    float result = FullTableBased.getS(P, H);

                                    ctx.getSource().sendSuccess(
                                            () -> Component.literal("Entropy (S): " + result),
                                            false
                                    );
                                    return 1;
                                })
                        )
                );
    }
    private static LiteralArgumentBuilder<CommandSourceStack> getHCommand() {
        return Commands.literal("getH")
                .then(Commands.argument("P", FloatArgumentType.floatArg(0))
                        .then(Commands.argument("S", FloatArgumentType.floatArg(0))
                                .executes(ctx -> {
                                    float P = FloatArgumentType.getFloat(ctx, "P");
                                    float S = FloatArgumentType.getFloat(ctx, "S");

                                    float result = FullTableBased.getH(P, S);

                                    ctx.getSource().sendSuccess(
                                            () -> Component.literal("Enthalpy (H): " + result),
                                            false
                                    );
                                    return 1;
                                })
                        )
                );
    }

    private static LiteralArgumentBuilder<CommandSourceStack> getTCommand() {
        return Commands.literal("getT")
                .then(Commands.argument("P", FloatArgumentType.floatArg(0))
                        .then(Commands.argument("H", FloatArgumentType.floatArg(0))
                                .executes(ctx -> {
                                    float P = FloatArgumentType.getFloat(ctx, "P");
                                    float H = FloatArgumentType.getFloat(ctx, "H");

                                    float result = FullTableBased.getT(P, H);

                                    ctx.getSource().sendSuccess(
                                            () -> Component.literal("Temperature (T): " + result),
                                            false
                                    );
                                    return 1;
                                })
                        )
                );
    }
    private static LiteralArgumentBuilder<CommandSourceStack> getXCommand() {
        return Commands.literal("getX")
                .then(Commands.argument("P", FloatArgumentType.floatArg(0))
                        .then(Commands.argument("H", FloatArgumentType.floatArg(0))
                                .executes(ctx -> {
                                    float P = FloatArgumentType.getFloat(ctx, "P");
                                    float H = FloatArgumentType.getFloat(ctx, "H");

                                    float result = FullTableBased.getX(P, H);

                                    ctx.getSource().sendSuccess(
                                            () -> Component.literal("Vapor Quality (X): " + result),
                                            false
                                    );
                                    return 1;
                                })
                        )
                );
    }
    private static int executeIsotropesCommand(CommandContext<CommandSourceStack> ctx) {
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

            Plotting.plotIsentropesPT(
                    "Water Isentropes (P–T)",
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
