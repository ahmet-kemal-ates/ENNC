% Run
function [obj, bestIndividual, bestFitness, history, convergenceTime, convergenceIter] = Run(obj)
% Run: Run the GA algorithm.
% The algorithm stop if:
%	1. Both all the global bests or their fitness values do not change significantly for consecutiveTrueTh number of iterations/generations.
%	2.The maximum number of iterations/generations is reached.
%
% Input:
% obj: GA class object
%
% Output:
% obj: GA class object
% bestElite: best parameters set. It represents the solution to the optimization problem
% bestFitness: fitness related to the best parameters set
% history: Evolution of the fitness over the iterations/generations
% convergenceTime: Time required for finding a solution
% convergenceIter: Iteration/generation in which the algorithms stopped

% Import Library
import heuristic_optimizer.DE.*

% Start global timer
timer = tic;

% Initialize consecutiveTrue fr stop condition
consecutiveTrue = 0;

% Initialize fitness history
history = zeros(obj.generalOptions.maxIteration, 1);

% Optimization core
for iter = 1:obj.generalOptions.maxIteration
    % Start iteration timer
    iterTime = tic;
    
    % Store best fitness
    history(iter) = min([obj.best.fitness]);
    
    % Verify Stop Condition
    if obj.StopCondition()
        % Increment consecutiveTrue counter if the StopCondition is verified
        consecutiveTrue = consecutiveTrue + 1;
    else
        % reset counter
        consecutiveTrue = 0;
    end
    
    % If the counter reached the threshold stop the algorithm
    if consecutiveTrue > obj.generalOptions.consecutiveTrueTh
        break;
    else
        % Save gBest for the evaluation of the stop condition
        obj.old_best = obj.best;
    end
    
    % Run the CallbackPre at the beginning of the iteration/generation
    if ~isempty(obj.CallbackPre)
        obj = obj.CallbackPre(obj, iter);
    end
    
    % Evolve Population
    obj = obj.EvolvePopulation();
    
    % Update Elites
    obj = obj.UpdateBest();
    
    % Run the CallbackPost at the end of the iteration/generation
    if ~isempty(obj.CallbackPost)
        obj = obj.CallbackPost(obj, iter);
    end
    
    % Print results
    if obj.verboseFlag
        iterTime = toc(iterTime);
        fprintf('Iter: %d/%d \t Best Fitness: %d \t Stop Condition: %d/%d \t Iter Time: %f \n', ...
            iter, obj.generalOptions.maxIteration, obj.best.fitness, consecutiveTrue, obj.generalOptions.consecutiveTrueTh, iterTime);
    end
end

% End of optimization

% Save best
bestIndividual = obj.best.chromosome;
bestFitness = obj.best.fitness;

% clear history
history(iter:end) = [];

% Save convergence information
convergenceIter = iter;
convergenceTime = toc(timer);
end