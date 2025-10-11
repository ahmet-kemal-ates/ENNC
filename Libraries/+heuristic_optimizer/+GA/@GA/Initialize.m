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
import heuristic_optimizer.GA.*

% Create the population
for p = 1:obj.generalOptions.numIndividuals
    % Call the Particle constructor
    obj.population(p,1) = Individual(obj.numParameters, obj.generalOptions, obj.GAOptions);
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
[~, sortedIndices] = sort([obj.population.fitness]);
obj.population = obj.population(sortedIndices);
for e=1:obj.GAOptions.numElites
    obj.elite(e) = struct('chromosome', obj.population(e).chromosome, 'fitness', obj.population(e).fitness);
end

% Initialize old_gBest for stopping condition
obj.old_elite = obj.elite;
end