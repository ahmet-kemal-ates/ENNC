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

function stopFlag = StopCondition(obj)
% StopCondition: Verify Stop Condition
%
% Input:
% obj: GA class object
%
% Output:
% stopFlag: Binary flag. 
%			stopFlag=1 either if all the global bests and the related fitness values 
%			did not change significantly from the previous iteration/generation.
%			stopFlag=0 otherwise.

% Import Library
import heuristic_optimizer.GA.*

% Initialize flag
stopFlag = false;

% Initialize vector containing the distance between old and new elites
deltaElites = zeros(obj.GAOptions.numElites,1);
deltaFitness = zeros(obj.GAOptions.numElites,1);

% Get the indices of binary variables
binaryFlag = obj.generalOptions.binaryFlag == 1;

% evaluate particle and fitness variations for each gBest
for e=1:obj.GAOptions.numElites
	% Evaluate distance for real parameters
	deltaRealParameters = pdist2(obj.old_elite(e).chromosome(~binaryFlag)', obj.elite(e).chromosome(~binaryFlag)');
	% Evaluate distance for binary variables
	deltaBinaryParameters = pdist2(obj.old_elite(e).chromosome(binaryFlag)', obj.elite(e).chromosome(binaryFlag)', 'hamming');
		
	% Handle if there are empty variables
	delta = [deltaRealParameters, deltaBinaryParameters];
	delta(isempty(delta) | isnan(delta)) = 0;
	
	% Store the distance
	deltaElites(e) = sum(delta);
	
	% Evaluate the absolute difference among the previous and the actual fitness values
	deltaFitness(e) = abs(obj.old_elite(e).fitness - obj.elite(e).fitness); 
end

% Check if all the stop condition are verified for all the gBest
if (prod(deltaElites < obj.generalOptions.stopThreshold) || prod(deltaFitness < obj.generalOptions.fitnessStopThreshold))
	stopFlag = true;
end

end