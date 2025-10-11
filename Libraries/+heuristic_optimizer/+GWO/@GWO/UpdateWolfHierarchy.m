% Update herd hierarcy
function obj = UpdateWolfHierarchy(obj)
import heuristic_optimizer.GWO.*

% sort herd with respect to fitness
[~, sortedIndices] = sort([obj.herd.fitness]);

for w=sortedIndices
    if obj.herd(w).fitness < obj.alphaWolf.fitness
        % Update Alpha wolf
        obj.alphaWolf.position = obj.herd(w).position;
        obj.alphaWolf.fitness = obj.herd(w).fitness;
    elseif obj.herd(w).fitness < obj.betaWolf.fitness
        % Update Beta wolf
        obj.betaWolf.position = obj.herd(w).position;
        obj.betaWolf.fitness = obj.herd(w).fitness;
    elseif obj.herd(w).fitness < obj.deltaWolf.fitness
        % Update Delta wolf
        obj.deltaWolf.position = obj.herd(w).position;
        obj.deltaWolf.fitness = obj.herd(w).fitness;
    end
end