% Update Elites
function obj = UpdateBest(obj)
import heuristic_optimizer.DE.*

%% Sort Population
[~, bestIndex] = min([obj.population.fitness]);

%% Evolve population
% Elite
if obj.population(bestIndex).fitness < obj.best.fitness
    obj.best.chromosome = obj.population(bestIndex).chromosome;
    obj.best.fitness = obj.population(bestIndex).fitness;
end
end