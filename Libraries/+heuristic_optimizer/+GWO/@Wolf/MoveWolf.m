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

function obj = MoveWolf(obj, a, alpha_position, beta_position, delta_position, p_Fitness, p_fitnessOptions)
% UpdateParticle: Perform a PSO evaluation step
%
% Input:
% obj: Wolf class object
% p_Fitness: Handle to the fitness function
% p_fitnessOptions: struct containing the further options and variables required for evaluating the fitness.
%
% Output:
% obj: Wolf class object 

maxDelta = obj.maxDelta;

% Follow the Alpha wolf
A_alpha = 2.*a.*rand(obj.numParameters,1)-a;
C_alpha = 2.*rand(obj.numParameters,1);

D_alpha = abs(C_alpha.*alpha_position - obj.position);
shift = A_alpha.*D_alpha;
shift(shift > maxDelta) = maxDelta;
shift(shift < -maxDelta) = -maxDelta;
X_alpha = alpha_position - shift;

% Follow the Beta wolf
A_beta = 2.*a.*rand(obj.numParameters,1)-a;
C_beta = 2.*rand(obj.numParameters,1);

D_beta = abs(C_beta.*beta_position - obj.position);
shift = A_beta.*D_beta;
shift(shift > maxDelta) = maxDelta;
shift(shift < -maxDelta) = -maxDelta;
X_beta = beta_position - shift;

% Follow the Delta wolf
A_delta = 2.*a.*rand(obj.numParameters,1)-a;
C_delta = 2.*rand(obj.numParameters,1);

D_delta = abs(C_delta.*delta_position - obj.position);
shift = A_delta.*D_delta;
shift(shift > maxDelta) = maxDelta;
shift(shift < -maxDelta) = -maxDelta;
X_delta = delta_position - shift;

% move wolf
obj.position = (X_alpha + X_beta + X_delta)/3;

% Update Fitness
obj.fitness = p_Fitness(obj.position, p_fitnessOptions);

end 