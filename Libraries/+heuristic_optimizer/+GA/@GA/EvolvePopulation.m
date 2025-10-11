% Update Swarm
function obj = EvolvePopulation(obj)
    import heuristic_optimizer.GA.*
    
    %% Sort Population
    [~, sortedIndices] = sort([obj.population.fitness]); 
    obj.population = obj.population(sortedIndices);
    
    %% Evolve population       
    % Apply Crossover
    availableRanks = 1:obj.generalOptions.numIndividuals;
    for n=1:2:2*obj.numCrossover
        % K tournament
        k = min(length(availableRanks), obj.k);
        shuffledRanks = availableRanks(randperm(length(availableRanks), k));
        selectedRank = sort(shuffledRanks);
        
        % Crossover
        [offspring_1, offspring_2] = ...
            obj.GAOptions.Crossover(obj.population(selectedRank(1)), obj.population(selectedRank(2)));
        obj.population(selectedRank(1)) = offspring_1;
        obj.population(selectedRank(2)) = offspring_2;
        % Delete selected index
        availableRanks(availableRanks == selectedRank(1) | availableRanks == selectedRank(2)) = [];
    end
    
    % Apply Mutation
    availableRanks = 1:obj.generalOptions.numIndividuals;
    for n=1:obj.numMutation
        % K tournament
        k = min(length(availableRanks), obj.k);
        shuffledRanks = availableRanks(randperm(length(availableRanks), k));
        selectedRank = min(shuffledRanks);
        
        % Mutate
        offspring = obj.GAOptions.Mutation(obj.population(selectedRank));
        obj.population(selectedRank) = offspring;
        % Delete selected index
        availableRanks(availableRanks == selectedRank) = [];
    end
    
    % Elite
    for e=1:obj.GAOptions.numElites
        obj.population(e).chromosome = obj.elite(e).chromosome;
    end
    
    %% Update Fitness
    localPopulation = obj.population;
    localFitness = obj.Fitness;
    localFitnessOptions = obj.fitnessOptions;
    parforArg = feature('numcores')*obj.generalOptions.parallelFlag;
    parfor (p = 1:obj.generalOptions.numIndividuals, parforArg)
        localPopulation(p) = localPopulation(p).UpdateFitness(localFitness, localFitnessOptions);
    end
    obj.population = localPopulation;
end