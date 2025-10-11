% Update Herd
function obj = UpdateHerd(obj)
    import heuristic_optimizer.GWO.*
    
    % Update the herd
    localHerd = obj.herd;
    localFitnessFunction = obj.Fitness;
    localFitnessOptions = obj.fitnessOptions;
    localA = obj.a;

    alpha_position = obj.alphaWolf.position;
    beta_position = obj.betaWolf.position;
    delta_position = obj.deltaWolf.position;
    
    parforArg = feature('numcores')*obj.generalOptions.parallelFlag;
    parfor (w=1:obj.generalOptions.numIndividuals, parforArg) 
        localHerd(w) = localHerd(w).MoveWolf(localA, alpha_position, beta_position, delta_position, localFitnessFunction, localFitnessOptions); 
    end
    obj.herd = localHerd;       
end