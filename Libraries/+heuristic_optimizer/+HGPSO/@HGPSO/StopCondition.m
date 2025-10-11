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
% obj: HGPSO class object
%
% Output:
% stopFlag: Binary flag. 
%			stopFlag=1 either if all the global bests and the related fitness values 
%			did not change significantly from the previous iteration/generation.
%			stopFlag=0 otherwise.

% Import Library
import heuristic_optimizer.HGPSO.*

% Initialize flag
stopFlag = false;

% Initialize vector containing the distance between old and new
% gBests
deltaBest = zeros(obj.generalOptions.numBests,1);
deltaFitness = zeros(obj.generalOptions.numBests,1);

% Get the indices of binary variables
binaryFlag = obj.generalOptions.binaryFlag == 1;

% evaluate particle and fitness variations for each gBest
for b=1:obj.generalOptions.numBests
	% Evaluate distance for real parameters
	deltaRealParameters = pdist2(obj.old_gBest(b).position(~binaryFlag)', obj.gBest(b).position(~binaryFlag)');
	% Evaluate distance for binary variables
	deltaBinaryParameters = pdist2(obj.old_gBest(b).position(binaryFlag)', obj.gBest(b).position(binaryFlag)', 'hamming');
		
	% Handle if there are empty variables
	delta = [deltaRealParameters, deltaBinaryParameters];
	delta(isempty(delta) | isnan(delta)) = 0;
	
	% Store the distance
	deltaBest(b) = sum(delta);
	
	% Evaluate the absolute difference among the previous and the actual fitness values
	deltaFitness(b) = abs(obj.gBest(b).fitness - obj.old_gBest(b).fitness); 
end

% Check if all the stop condition are verified for all the gBest
if (prod(deltaBest < obj.generalOptions.stopThreshold) || prod(deltaFitness < obj.generalOptions.fitnessStopThreshold))
	stopFlag = true;
end

end