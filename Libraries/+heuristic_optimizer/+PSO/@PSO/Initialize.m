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

function obj = Initialize(obj, initialSwarm)
% Initialize: Initialize the swarm of the HGPSO optimizer.
%
% Input:
% obj: HGPSO class object
% initialSwarm: Optional custom initialization. NxM matrix, where each column refers to one individual. N is the number of parameters and M is the number of individuals. 
%
% Output:
% obj: HGPSO class object

% Import library
import heuristic_optimizer.PSO.*

% Create the swarm
for p = 1:obj.generalOptions.numIndividuals
	% Call the Particle constructor
	obj.swarm(p,1) = Particle(obj.numParameters, obj.generalOptions, obj.PSOOptions);   
end
	
% Default Initialization
if ~exist('initialSwarm', 'var')
    % save local variables
    localSwarm = obj.swarm;
    localInitializationMode = obj.generalOptions.initializationMode;
    localInitializationRange = obj.generalOptions.initializationRange;
    localFitness = obj.Fitness;
    localFitnessOptions = obj.fitnessOptions;

    parforArg = feature('numcores')*obj.generalOptions.parallelFlag;
    parfor (p = 1:obj.generalOptions.numIndividuals, parforArg)
        % Call the ParticleClass Initialize method 
        localSwarm(p) = localSwarm(p).Initialize(localInitializationMode, localInitializationRange, localFitness, localFitnessOptions);
    end
    obj.swarm = localSwarm;
% Custom Initialization
else
		% save local variables
		localSwarm = obj.swarm;
		localNumParameters = obj.numParameters;
		localFitness = obj.Fitness;
		localFitnessOptions = obj.fitnessOptions;
        
        parforArg = feature('numcores')*obj.generalOptions.parallelFlag;
		parfor (p = 1:obj.generalOptions.numIndividuals, parforArg)
			% Get the p-th position
			localSwarm(p).position = initialSwarm(:, p);
		     
			% Initialize the velocity
			localSwarm(p).velocity = zeros(localNumParameters, 1);
			
			% Evaluate the fitness
			localSwarm(p).fitness = localFitness(localSwarm(p).position, localFitnessOptions);
			
			% Set the personal best to the actual position and the actual fitness
			localSwarm(p).pBest = localSwarm(p).position;
			localSwarm(p).pBestFitness = localSwarm(p).fitness;
			
			% Set the global best to the actual position of the individual
			localSwarm(p).gBest = localSwarm(p).position;
			localSwarm(p).subSwarmIndex = 1;
		end
		obj.swarm = localSwarm;
end

% Evaluate the gBests and set the subswarm membership

% sort swarm with respect to fitness
[~, sortedIndices] = sort([obj.swarm.fitness]); 
obj.swarm = obj.swarm(sortedIndices);
for b=1:obj.generalOptions.numBests
	obj.gBest(b) = struct('position', obj.swarm(b).position, 'fitness', obj.swarm(b).fitness, 'emptyCount', 0);
end

% Assign subswarm membership
for p = 1:obj.generalOptions.numIndividuals
	% Evaluate Eucludian distance
    delta = pdist2([obj.gBest.position]', obj.swarm(p).position');
	% Get the closest gBest
	[~, subSwarmIndex] = min(delta);
	% Assign subswarm and related global best
	obj.swarm(p).gBest = obj.gBest(subSwarmIndex).position;
	obj.swarm(p).subSwarmIndex = subSwarmIndex;
end

% Initialize old_gBest for stopping condition
obj.old_gBest = obj.gBest;
end