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

function obj = Initialize(obj, mode, range, p_Fitness, p_fitnessOptions)            
% Initialize: Initialize the particle
%
% Input:
% obj: Partcle class object
% mode: type of the selected initialization
% range: range of the initialization. Vector of Nx2 elements. first column is the lower range and the second column is the upper range. 
% p_Fitness: Handle to the fitness function
% p_fitnessOptions: struct containing the further options and variables required for evaluating the fitness.
%
% Output:
% obj: Particle class object 
   
% Identify the selected initialization mode and initialize the position of the individual correspondingly 
switch mode
	case 'zero'
		obj.position = zeros(obj.numParameters, 1);

	case 'uniform'
		obj.position = range(:,1) + abs(range(:,2)-range(:,1)).*rand(obj.numParameters, 1);

	case 'normal'
		avg = mean(range, 2).*ones(obj.numParameters, 1);
		sigma = abs(range(:,2)-range(:,1))/4.*ones(obj.numParameters, 1);
		obj.position = avg+sigma.*randn(obj.numParameters, 1);
        
        above_2sigma_index = obj.position>avg + 2*sigma;
        below_2sigma_index = obj.position<avg - 2*sigma;
		obj.position(above_2sigma_index)=avg(above_2sigma_index)+2*sigma(above_2sigma_index);
		obj.position(below_2sigma_index)=avg(below_2sigma_index)-2*sigma(below_2sigma_index);           
end

% Penalty on violation of boundaries
obj.penalty = sum(-(obj.position(obj.position<obj.lb) - obj.lb(obj.position<obj.lb))) + ...
	sum((obj.position(obj.position>obj.Ub) - obj.Ub(obj.position>obj.Ub)));

% Evaluate Fitness
obj.fitness = p_Fitness(obj.position, p_fitnessOptions) + obj.penalty;
end