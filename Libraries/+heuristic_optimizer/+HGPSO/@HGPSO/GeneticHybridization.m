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
% obj: HGPSO class object
%
% Output:
% obj: HGPSO class object

% Import library
import heuristic_optimizer.HGPSO.*

% Get k value for the K-tournament selection
k = max(2,round(obj.GAOptions.kRate*obj.generalOptions.numIndividuals));
% Get the number of crossover
numCrossover = min(floor(obj.generalOptions.numIndividuals/2), round(obj.GAOptions.crossoverRate*obj.generalOptions.numIndividuals/2));
% Get the number of mutations
numMutation = min(floor(obj.generalOptions.numIndividuals/2), round(obj.GAOptions.mutationRate*obj.generalOptions.numIndividuals));

% Sort swarm with respect to fitness
[~, sortedIndices] = sort([obj.swarm.fitness]); 
%obj.swarm = obj.swarm(sortedIndices);

% Select particles to substitute in the second half of the
% swarm
indicesForSubstitution = sortedIndices(...
    round(obj.generalOptions.numIndividuals/2) + ...
    randperm(round(obj.generalOptions.numIndividuals/2), 2*numCrossover+numMutation) ...
    );

% Apply Crossover
availableRanks = 1:obj.generalOptions.numIndividuals;
for n=1:2:2*numCrossover
	% K tournament
	shuffledRanks = availableRanks(randperm(length(availableRanks), k));
    selectedRank = sort(shuffledRanks);
	selectedIndex = sortedIndices(selectedRank(1:2));

	% Crossover
    if obj.GAOptions.worstFlag
        offspringIndex_1 = indicesForSubstitution(n);
        offspringIndex_2 = indicesForSubstitution(n+1);
    else
        offspringIndex_1 = selectedIndex(1);
        offspringIndex_2 = selectedIndex(2);
        indicesForSubstitution(n) = selectedIndex(1);
        indicesForSubstitution(n+1) = selectedIndex(2);
    end

	[obj.swarm(offspringIndex_1), obj.swarm(offspringIndex_2)] = ...
		obj.GAOptions.Crossover(obj.swarm(selectedIndex(1)), obj.swarm(selectedIndex(2)));

	% Delete selected index
	availableRanks(availableRanks == selectedRank(1) | availableRanks == selectedRank(2)) = [];
end

% Apply Mutation
availableRanks = 1:obj.generalOptions.numIndividuals;
for n=1:numMutation
	% K tournament
	shuffledRanks = availableRanks(randperm(length(availableRanks), k));
    selectedRank = min(shuffledRanks);
	selectedIndex = sortedIndices(selectedRank);

	% Mutate
    if obj.GAOptions.worstFlag
        mutedIndex = indicesForSubstitution(n+2*numCrossover);
    else
        mutedIndex = selectedIndex;
        indicesForSubstitution(n+2*numCrossover) = selectedIndex;
    end
	
    obj.swarm(mutedIndex) = obj.GAOptions.Mutation(obj.swarm(selectedIndex));

	% Delete selected index
	availableRanks(availableRanks == selectedRank) = [];
end

% Evaluate Fitness of the resulting particles and update the personal best
indicesForSubstitution=indicesForSubstitution(1:numMutation+2*numCrossover);
localSwarm = obj.swarm(indicesForSubstitution);
localFitness = obj.Fitness;
localFitnessOptions = obj.fitnessOptions;

parforArg = feature('numcores')*obj.generalOptions.parallelFlag;
parfor (n=1:length(indicesForSubstitution), parforArg)
    % Penalty on violation of boundaries
    violation_lb = localSwarm(n).position<localSwarm(n).lb;
    violation_Ub = localSwarm(n).position>localSwarm(n).Ub;
    localSwarm(n).penalty = sum(-(localSwarm(n).position(violation_lb) - localSwarm(n).lb(violation_lb))) + ...
                            sum(+(localSwarm(n).position(violation_Ub) - localSwarm(n).Ub(violation_Ub)));
    % Evaluate fitness
    localSwarm(n).fitness = localFitness(localSwarm(n).position, localFitnessOptions) + localSwarm(n).penalty; 
    if localSwarm(n).fitness < localSwarm(n).pBestFitness
        localSwarm(n).pBest = localSwarm(n).position;
        localSwarm(n).pBestFitness = localSwarm(n).fitness;
    end
end

obj.swarm(indicesForSubstitution) = localSwarm;
    
end