% Update Swarm
function obj = EvolvePopulation_rate(obj)
    import heuristic_optimizer.GA.*
    
    %% Sort Population
    [~, sortedIndices] = sort([obj.population.fitness]); 
    obj.population = obj.population(sortedIndices);
    
    % Evolve population
    %% Elite
    %obj.elite = obj.population(1:obj.numElites);
    for p = obj.GAOptions.numElites+1:obj.generalOptions.numIndividuals
        %% Crossover
        availableRanks = 1:obj.generalOptions.numIndividuals;
        availableRanks(p) = [];
        if rand > obj.GAOptions.crossoverRate
            % K tournament
            shuffledRanks = availableRanks(randperm(length(availableRanks), obj.k));
            selectedRank = min(shuffledRanks);
            
            % Crossover
            obj.population(p) = obj.population(p).Crossover(obj.population(selectedRank));
            
            % Delete selected index
            availableRanks(availableRanks == selectedRank) = [];
        end
        
        %% Mutation
        if rand > obj.GAOptions.mutationRate
            % Mutate
            obj.population(p) = obj.population(p).Mutation();
        end
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