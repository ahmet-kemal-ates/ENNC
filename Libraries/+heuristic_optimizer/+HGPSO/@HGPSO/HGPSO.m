% Copyright (c) 2017, 2018 Massimiliano Luzi
% University of Rome "La Sapienza"
%
% massimiliano.luzi@uniroma1.it
%
% This file is part of Heuristic Optimizers.
%
% Heuristic Optimizers is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% Heuristic Optimizers is distributed in the hope that it will be useful. 
% IT IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
% INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
% PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
% HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
% CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE 
% OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 
% See the GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with Heuristic Optimizers.If not, see<http://www.gnu.org/licenses/>.

classdef HGPSO
% Main class of the Hybrid Genetic Particle Swarm Optimization algorithm
    properties
        % Swarm
        swarm;
        numParameters;
        
        % Fitness
        Fitness;
        fitnessOptions;
        
        % Global bests
        gBest;
        old_gBest;
                
        % General Options
        generalOptions;
        
        % PSO Options
        PSOOptions;
        
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
        function obj = HGPSO(p_Fitness, p_numParameters, varargin)
			% HGPSO: Constructor of the HGPSO optimizer
			%
			% Input:
			% p_Fitness: Handle to the fitness function
			% p_numParameters: number of parameters to optimize
			% varargin: Supplementary options
			%
			% Output:
			% obj: HGPSO class object
			
			% Import library
            import heuristic_optimizer.HGPSO.*
            
			% Handle for the minimal number of required arguments
            if nargin <2
                error('The constructor must receive at leas two arguments: 1. The handle to the fitness function and 2. the number of parameters to optimize');
            end
            
            % Default algorithm parameters
            default_generalOptions = struct(...
                'parallelFlag',             false,      				... flag for the parallel evaluation
                'geneticFlag',              true,                       ... flag for the genetic hybridization
                'fullyInformedFlag',        false,                      ... flag for fully informed convergence PSO
                'guaranteedFlag',           true,      				    ... flag for guaranteed convergence PSO
                'binaryFlag',               zeros(p_numParameters, 1),	... flag for the Binary PSO
                'numIndividuals',           50,         				... num individuals of the pso
                'initializationMode',       'normal',					... type of initialization (zero, uniform, normal)
                'initializationRange',      [-1, 1],					... range of the initialization
                'lb',                       -inf(p_numParameters, 1),   ... no lower boundaries
                'Ub',                       inf(p_numParameters, 1),    ... no upper boundaries            
                'numBests',                 1,          				... Number of subswarms
                'killRatio',                0.05,       				... Ratio for killing the subswarm in case of few individuals
                'guaranteedDelta',          0.1,        				... max variation allowed for guaranteed convergence pso
                'maxVelocity',              0.1,        				... max variation allowed for velocity
                'maxIteration',             100,         				... max num of iteration
                'consecutiveTrueTh',        20,         				... threshold for the stop condition
                'stopThreshold',            1e-5,       				... threshold on the difference between old and new gBest
                'fitnessStopThreshold',     1e-5        				... threshold on the difference between old best fitness and new best fitness
                );
            
            % Default PSO parameters
            default_PSOOptions = struct(      ...
                'w',     0.7298,      ...   Inertial coefficient
                'c_p',   1.49618,     ...   Weight of personal best
                'c_g',   1.49618      ...   Weight of global best
                );

            % Default Genetic Hybridization parameters
            default_GAOptions = struct(             ...
                'worstFlag',        false,          ... Flag for substituting the worst particles, or the selected ones
                'kRate',            0.2,            ... Rate of population for the k value of K-tournament
                'crossoverRate',    0.1,            ... Rate of population to whose apply crossover
                'crossoverTh',      0.5,            ... Threshold for performing the crossover of feature
                'mutationRate',     0.1,            ... Rate of population to whose apply mutation
                'mutationTh',       0.75,           ... Threshold for performing the mutation of feature
                'mutationStd',      0.1,            ... Standard deviation of mutation
                'Crossover',        @obj.Crossover, ... Default function for crossover
                'Mutation',         @obj.Mutation   ... Default function for mutation
                );

            % Define the input parser
            p = inputParser;
            p.addRequired('p_Fitness');
            p.addRequired('p_numParameters');
            p.addParameter('generalOptions', default_generalOptions);
            p.addParameter('PSOOptions', default_PSOOptions);
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
            
            if ~isfield(obj.generalOptions, 'geneticFlag')
                obj.generalOptions.geneticFlag = default_generalOptions.geneticFlag;
            end
            
            if ~isfield(obj.generalOptions, 'fullyInformedFlag')
                obj.generalOptions.fullyInformedFlag = default_generalOptions.fullyInformedFlag;
            end
            
            if ~isfield(obj.generalOptions, 'guaranteedFlag')
                obj.generalOptions.guaranteedFlag = default_generalOptions.guaranteedFlag;
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
            
            if ~isfield(obj.generalOptions, 'numBests')
                obj.generalOptions.numBests = default_generalOptions.numBests1;
            end
            
            if ~isfield(obj.generalOptions, 'killRatio')
                obj.generalOptions.killRatio = default_generalOptions.killRatio;
            end
            
            if ~isfield(obj.generalOptions, 'guaranteedDelta')
                obj.generalOptions.maxDelta = default_generalOptions.guaranteedDelta;
            end
            
            if ~isfield(obj.generalOptions, 'maxVelocity')
                obj.generalOptions.maxVelocity = default_generalOptions.maxVelocity;
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
            
            % PSO Options
            obj.PSOOptions = p.Results.PSOOptions;
            
            % GA Options
            obj.GAOptions = p.Results.GAOptions;
			
            if obj.generalOptions.geneticFlag
                if ~isfield(obj.GAOptions, 'worstFlag')
                    obj.GAOptions.worstFlag = default_GAOptions.worstFlag;
                end
                
                if ~isfield(obj.GAOptions, 'kRate')
                    obj.GAOptions.kRate = default_GAOptions.kRate;
                end
                
                if ~isfield(obj.GAOptions, 'crossoverRate')
                    obj.GAOptions.crossoverRate = default_GAOptions.crossoverRate;
                end
                
                if ~isfield(obj.GAOptions, 'crossoverTh')
                    obj.GAOptions.crossoverTh = default_GAOptions.crossoverTh;
                end
                
                if ~isfield(obj.GAOptions, 'mutationRate')
                    obj.GAOptions.mutationRate = default_GAOptions.mutationRate;
                end
                
                if ~isfield(obj.GAOptions, 'mutationTh')
                    obj.GAOptions.mutationTh = default_GAOptions.mutationTh;
                end
                
                if ~isfield(obj.GAOptions, 'mutationStd')
                    obj.GAOptions.mutationStd = default_GAOptions.mutationStd;
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
            
            % Define swarm
            obj.swarm = Particle.empty;
                    
            % Define gBest and old_gBest
            obj.gBest = struct('position', [], 'fitness', inf, 'emptyCount', 0);
            obj.old_gBest = struct('position', [], 'fitness', inf, 'emptyCount', 0);
        end
    end
end