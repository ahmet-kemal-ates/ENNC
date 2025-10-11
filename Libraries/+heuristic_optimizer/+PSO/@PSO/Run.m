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

function [obj, bestPosition, bestFitness, history, convergenceTime, convergenceIter] = Run(obj)
% Run: Run the PSO algorithm. 
% The algorithm stop if:
%	1. Both all the global bests or their fitness values do not change significantly for consecutiveTrueTh number of iterations/generations.
%	2.The maximum number of iterations/generations is reached.
%
% Input:
% obj: PSO class object
%
% Output:
% obj: PSO class object
% bestPosition: best parameters set. It represents the solution to the optimization problem
% bestFitness: fitness related to the best parameters set
% history: Evolution of the fitness over the iterations/generations
% convergenceTime: Time required for finding a solution
% convergenceIter: Iteration/generation in which the algorithms stopped

% Import Library
import heuristic_optimizer.PSO.*

% Start wall clock
timer = tic;

% Initialize consecutiveTrue fr stop condition
consecutiveTrue = 0;

% Initialize fitness history
history = zeros(obj.generalOptions.maxIteration, 1);

% Optimization core
for iter = 1:obj.generalOptions.maxIteration
	% Start iteration timer
	iterTime = tic;
	
	% Store best fitness
	history(iter) = min([obj.gBest.fitness]);

	% Verify Stop Condition
	if obj.StopCondition()
		% Increment consecutiveTrue counter if the StopCondition is verified
		consecutiveTrue = consecutiveTrue + 1;
	else
		% reset counter
		consecutiveTrue = 0;
	end
	
	% If the counter reached the threshold stop the algorithm
	if consecutiveTrue > obj.generalOptions.consecutiveTrueTh
		break;
	else
		% Save gBest for the evaluation of the stop condition
		obj.old_gBest = obj.gBest;
	end
	
	% Run the CallbackPre at the beginning of the iteration/generation
	if ~isempty(obj.CallbackPre)
		obj = obj.CallbackPre(obj, iter);
	end
	
	% Update Swarm by applying the PSO equations
	obj = obj.UpdateSwarm();
	
	% Run the CallbackMid between the PSO and GA parts of the algorithm
	if ~isempty(obj.CallbackMid)
		obj = obj.CallbackMid(obj, iter);
    end
	
	% Update gBests
	obj = obj.UpdateGBest();

	% Run the CallbackPost at the end of the iteration/generation
	if ~isempty(obj.CallbackPost)
		obj = obj.CallbackPost(obj, iter);
	end

	% Print log
	if obj.verboseFlag
		iterTime = toc(iterTime);
		[~, bestGBestIndex] = min([obj.gBest.fitness]);
		fprintf('Iter: %d/%d \t Best Fitness: %d \t Stop Condition: %d/%d \t Iter Time: %f \n', ...
			iter, obj.generalOptions.maxIteration, obj.gBest(bestGBestIndex).fitness, consecutiveTrue, obj.generalOptions.consecutiveTrueTh, iterTime);
	end
end

% End of optimization

% Save gBest
[~, bestGBestIndex] = min([obj.gBest.fitness]);
bestPosition = obj.gBest(bestGBestIndex).position;
bestFitness = obj.gBest(bestGBestIndex).fitness;

% clear history in case of premature convergence
history(iter-1:end) = [];

% Save convergence information
convergenceIter = iter;
convergenceTime = toc(timer);

end