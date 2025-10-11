% Update Elites
function obj = UpdateElites(obj)
    import heuristic_optimizer.GA.*
    
    %% Sort Population
    [~, sortedIndices] = sort([obj.population.fitness]); 
    obj.population = obj.population(sortedIndices);
    
    %% Evolve population    
    % Elite
    for e=1:obj.GAOptions.numElites
        if obj.population(e).fitness < obj.elite(e).fitness
            obj.elite(e).chromosome = obj.population(e).chromosome;
            obj.elite(e).fitness = obj.population(e).fitness;
        end
    end
end