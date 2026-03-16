package com.rae.formicapi.simulation.nodal.thermal;

import com.rae.formicapi.math.solvers.ConjugateGradient;
import com.rae.formicapi.math.solvers.LeastSquare;
import com.rae.formicapi.simulation.nodal.core.*;

import java.util.Arrays;

public class SteadyStateSolver {

    public static void solve(SimulationModel model) {

        int n = model.getNodes().size();
        int m = 0;



        SimulationContext ctx = new SimulationContext(n, true);

        double[] x_init = new double[n];
        int id = 0;
        for (Node node : model.getNodes()) {
            node.setId(id);
            x_init[id] = node.getValue();
            if (node instanceof UnknownNode){
                m++;
            }
            if (node instanceof FixedValueNode){
                ctx.matrix.set(id,id, 1);//node.getValue();
                ctx.rhs[id] = node.getValue();
            }
            id++;
        }

        for (PhysicsComponent c : model.getComponents())
            c.stamp(ctx);


        //System.out.println(ctx.matrix);
        //System.out.println(Arrays.toString(x_init));
        //System.out.println(Arrays.toString(ctx.rhs));

        double[] result =
                LeastSquare.solve(ctx.matrix, x_init,ctx.rhs, 5000, 1e-3f);


        for (Node node : model.getNodes())
            node.setValue(result[node.getId()]);
    }
}
