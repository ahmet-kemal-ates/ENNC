classdef GWO
    properties
        % Herd
        herd;
        numParameters;
        a;
        
        % Fitness
        Fitness;
        fitnessOptions;
        
        % Alpha, beta and delta wolfs
        alphaWolf;
        betaWolf;
        deltaWolf;
        
        % Stop condition
        old_alphaWolf;
        old_betaWolf;
        old_deltaWolf;
        
        % General Options
        generalOptions;
        
        % GWO Options
        GWOOptions;
        
        % GA Options
        GAOptions;
        
        % Callback Functions 
        CallbackPre;
        CallbackMid;
        CallbackPost;
        
        % Verbose
        verboseFlag;
    end
    
    methods
        % Constructor
        function obj = GWO(p_Fitness, p_numParameters, varargin)
            % GWO: Constructor of the GWO optimizer
			%
			% Input:
			% p_Fitness: Handle to the fitness function
			% p_numParameters: number of parameters to optimize
			% varargin: Supplementary options
			%
			% Output:
			% obj: GWO class object
			
			% Import library
            import heuristic_optimizer.GWO.*
            
			% Handle for the minimal number of required arguments
            if nargin <2
                error('The constructor must receive at leas two arguments: 1. The handle to the fitness function and 2. the number of parameters to optimize');
            end
            
            % Default algorithm parameters
            default_generalOptions = struct(...
                'parallelFlag',             false,      				... flag for the parallel evaluation
                'numIndividuals',           50,         				... num individuals of the pso
                'initializationMode',       'normal',					... type of initialization (zero, uniform, normal)
                'initializationRange',      [-1, 1],					... range of the initialization
                'lb',                       -inf(p_numParameters, 1),   ... no lower boundaries
                'Ub',                       inf(p_numParameters, 1),    ... no upper boundaries            
                'maxDelta',                 0.1,        				... max variation allowed for guaranteed convergence pso
                'maxIteration',             100,         				... max num of iteration
                'consecutiveTrueTh',        20,         				... threshold for the stop condition
                'stopThreshold',            1e-5,       				... threshold on the difference between old and new gBest
                'fitnessStopThreshold',     1e-5        				... threshold on the difference between old best fitness and new best fitness
                );
            
            % Default PSO parameters
            default_GWOOptions = struct(      ...
                'a_0',     2      ...   Inertial coefficient
                );

            % Default Genetic Hybridization parameters
            default_GAOptions = [];

            % Define the input parser
            p = inputParser;
            p.addRequired('p_Fitness');
            p.addRequired('p_numParameters');
            p.addParameter('generalOptions', default_generalOptions);
            p.addParameter('GWOOptions', default_GWOOptions);
            p.addParameter('GAOptions', default_GAOptions);
            p.addParameter('fitnessOptions', []);
            p.addParameter('CallbackPre', []);
            p.addParameter('CallbackMid', []);
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
                obj.generalOptions.parallelFlag = default_generalOptions.parallelFlag;
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
            
            if ~isfield(obj.generalOptions, 'maxDelta')
                obj.generalOptions.maxDelta = default_generalOptions.maxDelta;
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
                obj.generalOptions.fitnessStopThreshold = default_generalOptions.fitnessStopThreshol;
            end
            
            % PSO Options
            obj.GWOOptions = p.Results.GWOOptions;
            
            % GA Options
            obj.GAOptions = p.Results.GAOptions;
			
            if ~isempty(obj.GAOptions)
                if ~isfield(obj.GAOptions, 'kRate')
                    obj.GAOptions.kRate = 0.2;
                end
                
                if ~isfield(obj.GAOptions, 'crossoverRate')
                    obj.GAOptions.crossoverRate = 0.1;
                end
                
                if ~isfield(obj.GAOptions, 'crossoverTh')
                    obj.GAOptions.crossoverTh = 0.5;
                end
                
                if ~isfield(obj.GAOptions, 'mutationRate')
                    obj.GAOptions.mutationRate = 0.1;
                end
                
                if ~isfield(obj.GAOptions, 'mutationTh')
                    obj.GAOptions.mutationTh = 0.75;
                end
                
                if ~isfield(obj.GAOptions, 'mutationStd')
                    obj.GAOptions.mutationStd = 0.1;
                end
                
                if ~isfield(obj.GAOptions, 'Crossover')
                    obj.GAOptions.Crossover = @obj.Crossover;
                end
                
                if ~isfield(obj.GAOptions, 'Mutation')
                    obj.GAOptions.Mutation = @obj.Mutation;
                end
            end
            
            % Callback Options
            obj.CallbackPre = p.Results.CallbackPre;
            obj.CallbackMid = p.Results.CallbackMid;
            obj.CallbackPost = p.Results.CallbackPost;
            
            % Verbose
            obj.verboseFlag = p.Results.Verbose;
            
            % Define herd
            obj.herd = Wolf.empty;
                    
            % Define alpha, beta and delta wolfes
            obj.alphaWolf = struct('position', [], 'fitness', inf);
            obj.betaWolf = struct('position', [], 'fitness', inf);
            obj.deltaWolf = struct('position', [], 'fitness', inf);
            
            % old alpha wolf for stop confition
            obj.old_alphaWolf = struct('position', [], 'fitness', inf);
            obj.old_betaWolf = struct('position', [], 'fitness', inf);
            obj.old_deltaWolf = struct('position', [], 'fitness', inf);
        end
    end
end