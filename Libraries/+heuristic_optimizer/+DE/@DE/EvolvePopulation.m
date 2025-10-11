% Update Swarm
function obj = EvolvePopulation(obj)
    import heuristic_optimizer.DE.*
    
    localPopulation = obj.population;
    selected = cell(obj.generalOptions.numIndividuals,1);
    localFitness = obj.Fitness;
    localFitnessOptions = obj.fitnessOptions;
    
    for p = 1:obj.generalOptions.numIndividuals
        availableRanks = 1:obj.generalOptions.numIndividuals;
        availableRanks(p) = [];
        selectedRank = availableRanks(randperm(length(availableRanks), 3));
        selected{p} = localPopulation(selectedRank);
    end
    parforArg = feature('numcores')*obj.generalOptions.parallelFlag;
    parfor (p = 1:obj.generalOptions.numIndividuals, parforArg)
        localPopulation(p) = localPopulation(p).Evolve(selected{p}, localFitness, localFitnessOptions);
    end
    obj.population = localPopulation;
end