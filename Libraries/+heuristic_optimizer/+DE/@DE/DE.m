classdef DE
    properties
        % Population
        population;
        numParameters;
        
        % Fitness
        Fitness;
        fitnessOptions;
                
        % Elites
        best;
        old_best;
        
        % General Options
        generalOptions;
        
        % DE Options
        DEOptions;
        
        % Callback Function 
        CallbackPre;
        CallbackPost;
        
        % Verbose
        verboseFlag;
    end
    
    methods
        % Constructor
        function obj = DE(p_Fitness, p_numParameters, varargin)
            import heuristic_optimizer.DE.*
            
            if nargin <2
                error('The constructor must receive at leas two arguments: 1. The handle to the fitness function and 2. the number of paramters to optimize');
            end
            
            % Default algorithm parameters
            default_generalOptions = struct(...
                'parallelFlag',             false,                          ... flag for the parallel evaluation
                'binaryFlag',               zeros(p_numParameters, 1),      ... flag for the Binary GA
                'numIndividuals',           50,                             ... num individuals of the GA
                'initializationMode',       'normal',                       ... type of initialization (zero, uniform, normal)
                'initializationRange',      [-1, 1],                        ... range of the initialization
                'lb',                       -inf(p_numParameters, 1),       ... no lower boundaries
                'Ub',                       inf(p_numParameters, 1),        ... no upper boundaries            
                'maxIteration',             50,         ... max num of iteration
                'consecutiveTrueTh',        50,         ... threshold for the stop condition
                'stopThreshold',            1e-5,       ... threhsold on the difference between old and new gBest
                'fitnessStopThreshold',     1e-5        ... threshold on the difference between old best fitness and new best fitness
                );

            % Default Differential Evolution parameters
            default_DEOptions = struct(  ...
                'kRate',         0.3,                           ... kRate for K tournament used in GA hybridization
                'mutationRate',  0.1,                           ... Percentage of numIndividuals defining the number of mutation    
                'mutationTh',    0.8*ones(p_numParameters, 1),  ... Threshold for the gene to mutate
                'mutationStd',   0.2*ones(p_numParameters, 1),  ... Std for mutate the position
                'Mutation',      @obj.Mutation,                     ... Handler of the mutation function
                'crossoverRate', 0.8,                           ... Percentage of numIndividuals defining the number of crossovers  
                'crossoverTh',   0.5*ones(p_numParameters, 1),  ... Threshold for the gene to mutate  
                'Crossover',     @obj.Crossover                      ... Handler for the crossover function
                );

            % Define the input parser
            p = inputParser;
            p.addRequired('p_Fitness');
            p.addRequired('p_numParameters');
            p.addParameter('generalOptions', default_generalOptions);
            p.addParameter('DEOptions', default_DEOptions);
            p.addParameter('fitnessOptions', []);
            p.addParameter('CallbackPre', []);
            p.addParameter('CallbackPost', []);
            p.addParameter('Verbose', true);
           
            % Parse arguments
            p.parse(p_Fitness, p_numParameters, varargin{:});
            
            % Fitness Options
            obj.Fitness = p_Fitness;
            obj.fitnessOptions = p.Results.fitnessOptions;
            
            % Particle information
            obj.numParameters = p_numParameters;
            
            % General Options
            obj.generalOptions = p.Results.generalOptions;
            % Handle for undefined struct fields
            if ~isfield(obj.generalOptions, 'parallelFlag')
                obj.generalOptions.parallelFlag = default_generalOptions.parallelFla;
            end
            
            if ~isfield(obj.generalOptions, 'binaryFlag')
                obj.generalOptions.binaryFlag = default_generalOptions.binaryFlag;
            else
                assert(length(obj.generalOptions.binaryFlag)==p_numParameters, 'binaryFlag option should be a binary vector of lenght equal to the number of parameters.')
            end
            
            if ~isfield(obj.generalOptions, 'numIndividuals')
                obj.generalOptions.numIndividuals = default_generalOptions.numIndividuals;
            end
            
            if ~isfield(obj.generalOptions, 'initializationMode')
                obj.generalOptions.initializationMode =  default_generalOptions.initializationMode;
            end
            
            if ~isfield(obj.generalOptions, 'initializationRange')
                obj.generalOptions.initializationRange = default_generalOptions.initializationRange;
            end
                
            if ~isfield(obj.generalOptions, 'lb')
                obj.generalOptions.lb = default_generalOptions.lb;
            end
            
            if ~isfield(obj.generalOptions, 'Ub')
                obj.generalOptions.Ub = default_generalOptions.Ub;
            end
            
            if ~isfield(obj.generalOptions, 'maxIteration')
                obj.generalOptions.maxIteration = default_generalOptions.maxIteration;
            end
            if ~isfield(obj.generalOptions, 'consecutiveTrueTh')
                obj.generalOptions.consecutiveTrueTh = default_generalOptions.consecutiveTrueTh;
            end
            
            if ~isfield(obj.generalOptions, 'stopThreshold')
                obj.generalOptions.stopThreshold = default_generalOptions.stopThreshold;
            end
            
            if ~isfield(obj.generalOptions, 'fitnessStopThreshold')
                obj.generalOptions.fitnessStopThreshold = default_generalOptions.fitnessStopThreshold;
            end
            
            % DE Options
            obj.DEOptions = p.Results.DEOptions;
            if ~isfield(obj.DEOptions, 'kRate')
                obj.DEOptions.kRate = default_DEOptions.kRate;
            end
            if ~isfield(obj.DEOptions, 'mutationRate')
                obj.DEOptions.mutationRate = default_DEOptions.mutationRate;
            end
            if ~isfield(obj.DEOptions, 'mutationTh')
                obj.DEOptions.mutationTh = default_DEOptions.mutationTh;
            end
            if ~isfield(obj.DEOptions, 'mutationStd')
                obj.DEOptions.mutationStd = default_DEOptions.mutationStd;
            end
            if ~isfield(obj.DEOptions, 'Mutation')
                obj.DEOptions.Mutation = @obj.Mutation;
            end
            if ~isfield(obj.DEOptions, 'crossoverRate')
                obj.DEOptions.crossoverRate = default_DEOptions.crossoverRate;
            end
            if ~isfield(obj.DEOptions, 'crossoverTh')
                obj.DEOptions.crossoverTh = default_DEOptions.crossoverTh;
            end
            if ~isfield(obj.DEOptions, 'Crossover')
                obj.DEOptions.Crossover = @obj.Crossover;
            end
            
            % Callback Options
            obj.CallbackPre = p.Results.CallbackPre;
            obj.CallbackPost = p.Results.CallbackPost;
            
            % Verbose
            obj.verboseFlag = p.Results.Verbose;
            
            % Define population
            obj.population = Individual.empty;

            % Define best and old_best
            obj.best = struct('chromosome', [], 'fitness', inf);
            obj.old_best = struct('chromosome', [], 'fitness', inf);
        end
    end
end