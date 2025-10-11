% Initialization
function obj = Initialize(obj, initialPopulation)
% Initialize: Initialize the swarm of the GA optimizer.
%
% Input:
% obj: GA class object
% initialPopulation: Optional custom initialization. NxM matrix, where each column refers to one individual. N is the number of parameters and M is the number of individuals.
%
% Output:
% obj: GA class object

% Import library
import heuristic_optimizer.DE.*

% Create the population
for p = 1:obj.generalOptions.numIndividuals
    % Call the Particle constructor
    obj.population(p,1) = Individual(obj.numParameters, obj.generalOptions, obj.DEOptions);
end

% Default Initialization
if ~exist('initialPopulation', 'var')
    % save local variables
    localPopulation = obj.population;
    localInitializationMode = obj.generalOptions.initializationMode;
    localInitializationRange = obj.generalOptions.initializationRange;
    localFitness = obj.Fitness;
    localFitnessOptions = obj.fitnessOptions;
    
    parforArg = feature('numcores')*obj.generalOptions.parallelFlag;
    parfor (p = 1:obj.generalOptions.numIndividuals, parforArg)
        % Call the ParticleClass Initialize method
        localPopulation(p) = localPopulation(p).Initialize(localInitializationMode, localInitializationRange, localFitness, localFitnessOptions);
    end
    obj.population = localPopulation;
    % Custom Initialization
else
    % save local variables
    localPopulation = obj.population;
    localFitness = obj.Fitness;
    localFitnessOptions = obj.fitnessOptions;
    
    parforArg = feature('numcores')*obj.generalOptions.parallelFlag;
    parfor (p = 1:obj.generalOptions.numIndividuals, parforArg)
        % Get the p-th chromosome
        localPopulation(p).chromosome = initialPopulation(:, p);
        
        % Evaluate the fitness
        localPopulation(p).fitness = localFitness(localPopulation(p).position, localFitnessOptions);
    end
    obj.population = localPopulation;
end

% Evaluate the elites
% sort population with respect to fitness
[~, bestIndex] = min([obj.population.fitness]);

obj.best = struct('chromosome', obj.population(bestIndex).chromosome, 'fitness', obj.population(bestIndex).fitness);

% Initialize old_gBest for stopping condition
obj.old_best = obj.best;
end