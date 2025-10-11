% Copyright (c) 2017, 2018 Massimiliano Luzi
% University of Rome "La Sapienza"
%
% massimiliano.luzi@uniroma1.it
%
% This file is part of Heuristic Optimizers.
%
% HeuristicLibrary is free software: you can redistribute it and/or modify
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

function obj = UpdateGBest(obj)
% UpdateGBest: Update the globl bests positions and the particle subswarm memberships
%
% Input:
% obj: HGPSO class object
%
% Output:
% obj: HGPSO class object

% Import Library
import heuristic_optimizer.HGPSO.*

% Get kill Threshold for killing the less populated subswarms
killThreshold = obj.generalOptions.killRatio*obj.generalOptions.numIndividuals;

% Update each gBests considering the particles belonging to its
% subswarm
for b=1:obj.generalOptions.numBests      
	% Get the subswarm
	subSwarm = obj.swarm([obj.swarm.subSwarmIndex] == b);

	% Get the particle with the lowest fitness in the current
	% subswarm
	[local_gBestFitness, local_gBestIndex] = min([subSwarm.fitness]);

	% Check if the subswarm has sufficient particles
	if length(subSwarm) > killThreshold
		obj.gBest(b).emptyCount = 0;

		% Update the current gBest if it exists a better
		% particle
		if local_gBestFitness < obj.gBest(b).fitness
			obj.gBest(b).position = subSwarm(local_gBestIndex).position;
			obj.gBest(b).fitness = local_gBestFitness;
		end
	% If the subswarm is too few populated
	else
		% Increment counter
		obj.gBest(b).emptyCount = obj.gBest(b).emptyCount + 1;

		% If counter is greater than a threshold reinitialize
		% the current gBest
		if obj.gBest(b).emptyCount > 5     
			obj.gBest(b).emptyCount = 0;
			obj.gBest(b).position = obj.swarm(b).position;
			obj.gBest(b).fitness = obj.swarm(b).fitness;
		% Else check if the gBest has to be update
		elseif ~isempty(subSwarm)
			if local_gBestFitness < obj.gBest(b).fitness
				obj.gBest(b).position = subSwarm(local_gBestIndex).position;
				obj.gBest(b).fitness = local_gBestFitness;
			end
		end
	end
end

% Sort gBests
% [~, gBestSortedIndecies] = sort([obj.gBest.fitness]);
% obj.gBest = obj.gBest(gBestSortedIndecies);

% Update subSwarm membership
for p = 1:obj.generalOptions.numIndividuals
	% Evaluate closest global best
	delta = pdist2([obj.gBest.position]', obj.swarm(p).position');
	[~, subSwarmIndex] = min(delta);
	
	% copy ParticleClass
	individual = copy(obj.swarm(p));
	
	% Update gBest position and subswarm membership
	individual.gBest = obj.gBest(subSwarmIndex).position;
	individual.subSwarmIndex = subSwarmIndex;
	
	% Update actual individual
	obj.swarm(p) = individual;
end

end