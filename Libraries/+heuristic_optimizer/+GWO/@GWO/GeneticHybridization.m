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

function obj = GeneticHybridization(obj)
% GeneticHybridization: method performing the genetic hybridization of the algorithm.
%
% Input:
% obj: GWO class object
%
% Output:
% obj: GWO class object

% Import library
import heuristic_optimizer.GWO.*

% Get k value for the K-tournament selection
k = max(2,round(obj.GAOptions.kRate*obj.generalOptions.numIndividuals));
% Get the number of crossover
numCrossover = min(floor(obj.generalOptions.numIndividuals/2), round(obj.GAOptions.crossoverRate*obj.generalOptions.numIndividuals/2));
% Get the number of mutations
numMutation = min(floor(obj.generalOptions.numIndividuals/2), round(obj.GAOptions.mutationRate*obj.generalOptions.numIndividuals));

% Sort herd with respect to fitness
[~, sortedIndices] = sort([obj.herd.fitness]); 
obj.herd = obj.herd(sortedIndices);

% Select particles to substitute in the second half of the
% herd
indicesForSubstitution = round(obj.generalOptions.numIndividuals/2) +  randperm(round(obj.generalOptions.numIndividuals/2), 2*numCrossover+numMutation);

% Apply Crossover
availableIndeces = 1:obj.generalOptions.numIndividuals;
for n=1:2:2*numCrossover
	% K tournament
	shuffledIndeces = availableIndeces(randperm(length(availableIndeces), k));
	selectedIndex = sort(shuffledIndeces);

	% Crossover
	[obj.herd(indicesForSubstitution(n)), obj.herd(indicesForSubstitution(n+1))] = ...
		obj.GAOptions.Crossover(obj.herd(selectedIndex(1)), obj.herd(selectedIndex(2)));

	% Delete selected index
	availableIndeces(availableIndeces == selectedIndex(1) | availableIndeces == selectedIndex(2)) = [];
end

% Apply Mutation
availableIndeces = 1:obj.generalOptions.numIndividuals;
for n=1:numMutation
	% K tournament
	shuffledIndeces = availableIndeces(randperm(length(availableIndeces), k));
	selectedIndex = min(shuffledIndeces);

	% Mutate
	obj.herd(indicesForSubstitution(n+2*numCrossover)) = obj.GAOptions.Mutation(obj.herd(selectedIndex));

	% Delete selected index
	availableIndeces(availableIndeces == selectedIndex) = [];
end

% Evaluate Fitness of the resulting particles and update the personal best
indicesForSubstitution=indicesForSubstitution(1:numMutation+2*numCrossover);
localHerd = obj.herd(indicesForSubstitution);
localFitness = obj.Fitness;
localFitnessOptions = obj.fitnessOptions;

parforArg = feature('numcores')*obj.generalOptions.parallelFlag;
parfor (n=1:length(indicesForSubstitution), parforArg)
    % Penalty on violation of boundaries
    localHerd(n).penalty = sum(-(localHerd(n).position(localHerd(n).position<localHerd(n).lb) - localHerd(n).lb(localHerd(n).position<localHerd(n).lb))) + ...
        sum((localHerd(n).position(localHerd(n).position>localHerd(n).Ub) - localHerd(n).Ub(localHerd(n).position>localHerd(n).Ub)));
    % Evaluate fitness
    localHerd(n).fitness = localFitness(localHerd(n).position, localFitnessOptions) + localHerd(n).penalty; 
end

obj.herd(indicesForSubstitution) = localHerd;
    
end