% Initialization
function obj = Initialize(obj, initialHerd)
import heuristic_optimizer.GWO.*

obj.a = obj.GWOOptions.a_0;

% Instantiate the herd
for w = 1:obj.generalOptions.numIndividuals
    % Call the Particle constructor
    obj.herd(w,1) = Wolf(obj.numParameters, obj.generalOptions);
end

% Initialize the herd

% save local variables
localHerd = obj.herd;
localInitializationMode = obj.generalOptions.initializationMode;
localInitializationRange = obj.generalOptions.initializationRange;
localFitness = obj.Fitness;
localFitnessOptions = obj.fitnessOptions;


if ~exist('initialHerd', 'var')
    parforArg = feature('numcores')*obj.generalOptions.parallelFlag;
    parfor (w = 1:obj.generalOptions.numIndividuals, parforArg)
        localHerd(w) = localHerd(w).Initialize(localInitializationMode, localInitializationRange, localFitness, localFitnessOptions);
    end
    obj.herd = localHerd;
    % Custom initialization
else
    parforArg = feature('numcores')*obj.generalOptions.parallelFlag;
    parfor (w = 1:obj.generalOptions.numIndividuals, parforArg)
        % Get the w-th position
        localHerd(w).position = initialHerd(:, w);
        
        % Evaluate the fitness
        localHerd(w).fitness = localFitness(localHerd(w).position, localFitnessOptions);
    end
    obj.herd = localHerd;
end

% Find the alpha, beta and delta wolfes
% sort herd with respect to fitness
[~, sortedIndices] = sort([obj.herd.fitness]);

% Alpha wolf
alpha_index = sortedIndices(1);
obj.alphaWolf.position = obj.herd(alpha_index).position;
obj.alphaWolf.fitness = obj.herd(alpha_index).fitness;
% Beta wolf
beta_index = sortedIndices(2);
obj.betaWolf.position = obj.herd(beta_index).position;
obj.betaWolf.fitness = obj.herd(beta_index).fitness;
% Delta wolf
delta_index = sortedIndices(3);
obj.deltaWolf.position = obj.herd(delta_index).position;
obj.deltaWolf.fitness = obj.herd(delta_index).fitness;

% Initialize old_alphaWolf for stopping condition
obj.old_alphaWolf = obj.alphaWolf;
obj.old_betaWolf = obj.betaWolf;
obj.old_deltaWolf = obj.deltaWolf;
end