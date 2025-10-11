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
% obj: PSO class object
%
% Output:
% obj: PSO class object

% Import Library
import heuristic_optimizer.PSO.*

% Get the particle with the lowest fitness 
[local_gBestFitness, local_gBestIndex] = min([obj.swarm.fitness]);

% Update the current gBest if it exists a better
% particle
if local_gBestFitness < obj.gBest.fitness
    obj.gBest.position = obj.swarm(local_gBestIndex).position;
    obj.gBest.fitness = local_gBestFitness;
end

% Update gBest in particles
for p = 1:obj.generalOptions.numIndividuals
	% copy ParticleClass
	individual = copy(obj.swarm(p));
	
	% Update gBest position and subswarm membership
	individual.gBest = obj.gBest.position;
	individual.subSwarmIndex = 1;
	
	% Update actual individual
	obj.swarm(p) = individual;
end

end