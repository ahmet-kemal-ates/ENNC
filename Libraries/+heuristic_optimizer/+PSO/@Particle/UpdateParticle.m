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

function obj = UpdateParticle(obj, p_Fitness, p_fitnessOptions)
% UpdateParticle: Perform a PSO evaluation step
%
% Input:
% obj: Partcle class object
% p_Fitness: Handle to the fitness function
% p_fitnessOptions: struct containing the further options and variables required for evaluating the fitness.
%
% Output:
% obj: Partcle class object


% Evaluate Velocity

% Get the random coefficient
r_p = rand(obj.numParameters,1);
r_g = rand(obj.numParameters,1);
obj.velocity = obj.w*obj.velocity + ...
    obj.c_p.*r_p.*(obj.pBest - obj.position) + ...
    obj.c_g.*r_g.*(obj.gBest - obj.position);


% Clip the velocity if it is out of its maximum value
obj.velocity = min(obj.velocity, obj.maxVelocity);
obj.velocity = max(obj.velocity, obj.maxVelocity);

% Update position
binaryFlag = obj.binaryFlag == 1;

% Real variables
obj.position(~binaryFlag) = obj.position(~binaryFlag) + obj.velocity(~binaryFlag);

% Binary variables
r_b = rand(obj.numParameters,1);
sigmoidV = 1./(1+exp(-obj.velocity));
obj.position(binaryFlag & sigmoidV > r_b) = 1;
obj.position(binaryFlag & sigmoidV <= r_b) = 0;

% Penalty on violation of boundaries
violation_lb = obj.position<obj.lb;
violation_Ub = obj.position>obj.Ub;
obj.penalty = sum(-(obj.position(violation_lb) - obj.lb(violation_lb))) + ...
    sum(+(obj.position(violation_Ub) - obj.Ub(violation_Ub)));

% Update Fitness
obj.fitness = p_Fitness(obj.position, p_fitnessOptions) + obj.penalty;

% Update Personal Best
if obj.fitness < obj.pBestFitness
    obj.pBest = obj.position;
    obj.pBestFitness = obj.fitness;
end

end