% Run
function [obj, bestPosition, bestFitness, hystory, convergenceTime, convergenceIter] = Run(obj)
    import heuristic_optimizer.GWO.*

    % Start global timer
    timer = tic;

    % Initialize consecutiveTrue fr stop condition
    consecutiveTrue = 0;

    % Initialize fitness hystory
    hystory = zeros(obj.generalOptions.maxIteration, 1);

    % Optimization core
    for iter = 1:obj.generalOptions.maxIteration
        % Start iteration timer
        iterTime = tic;

        % Store best fitness
        hystory(iter) = obj.alphaWolf.fitness;

        % Verify Stop Condition
        if obj.StopCondition()
            consecutiveTrue = consecutiveTrue + 1;
        else
            consecutiveTrue = 0;
        end

        if consecutiveTrue > obj.generalOptions.consecutiveTrueTh
            break;
        else
            % Save gBest
            obj.old_alphaWolf = obj.alphaWolf;
        end
        
        % Run the CallbackPre at the beginning of the iteration/generation
        if ~isempty(obj.CallbackPre)
            obj = obj.CallbackPre(obj, iter);
        end
        
        % Reduce a
        obj.a = obj.GWOOptions.a_0*(1 - ((iter-1)/obj.generalOptions.maxIteration));

        % Update Herd
        obj = obj.UpdateHerd();

        % Generate new particles with genetic operators
        if ~isempty(obj.GAOptions)
            obj = obj.GeneticHybridization();
        end
        
        % Update Hierarhcy
        obj = obj.UpdateWolfHierarchy();
        
        % Run the CallbackPost at the end of the iteration/generation
        if ~isempty(obj.CallbackPost)
            obj = obj.CallbackPost(obj, iter);
        end
        
        % Print results
        iterTime = toc(iterTime);
        fprintf('Iter: %d/%d \n\t Alpha Fitness: %d \n\t Beta Fitness: %d \n\t Delta Fitness: %d \n\t Stop Condition: %d/%d \t Iter Time: %f \n', ...
            iter, obj.generalOptions.maxIteration, obj.alphaWolf.fitness, obj.betaWolf.fitness, obj.deltaWolf.fitness, consecutiveTrue, obj.generalOptions.consecutiveTrueTh, iterTime);
    end

    % End of optimization

    % Save Alpha Wolf
    bestPosition = obj.alphaWolf.position;
    bestFitness = obj.alphaWolf.fitness;

    % clear hystory
    hystory(iter:end) = [];

    % Save convergence information
    convergenceIter = iter;
    convergenceTime = toc(timer);
end