% Stop Condition
function stopFlag = StopCondition(obj)
    import heuristic_optimizer.GWO.*
    
    % Initialize flag
    stopFlag = false;

    % Evaluate differences in position and fitness for the alpha wolf
    deltaAlpha = pdist2(obj.old_alphaWolf.position', obj.alphaWolf.position');
    deltaFitnessAlpha = abs(obj.old_alphaWolf.fitness - obj.alphaWolf.fitness);
    % Evaluate differences in position and fitness for the beta wolf
    deltaBeta = pdist2(obj.old_betaWolf.position', obj.betaWolf.position');
    deltaFitnessBeta = abs(obj.old_alphaWolf.fitness - obj.alphaWolf.fitness);
    % Evaluate differences in position and fitness for the delta wolf
    deltaDelta = pdist2(obj.old_deltaWolf.position', obj.deltaWolf.position');
    deltaFitnessDelta = abs(obj.old_alphaWolf.fitness - obj.alphaWolf.fitness);
    
    % Check if all the stop condition are verified for each gBest
    deltaLeaders = [deltaAlpha, deltaBeta, deltaDelta];
    deltaFitness = [deltaFitnessAlpha, deltaFitnessBeta, deltaFitnessDelta];  
    if (prod(deltaLeaders < obj.generalOptions.stopThreshold) || prod(deltaFitness < obj.generalOptions.fitnessStopThreshold))
        stopFlag = true;
    end
end