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

function obj = Evolve(obj, agents, p_Fitness, p_fitnessOptions)
% Crossover: Apply crossover operator to the input individuals
% Scattring crossover. Each features is swapped among the two parents with a certain probability given by 1-crossoverTh
%
% Input:
% obj: Indivudal class object. It is the first parent
% partner: Individual class referring to the partner of crossover 
%
% Output:
% obj: Individual object referring to offspring resulting from the crossover operator

% Create the mask for applying the crossover. Where crossoverMask==1 the related features will be swapped
recombinationMask = round(...
                        rand(obj.numParameters, 1)<obj.crossoverTh | ...
                        randi(obj.numParameters, obj.numParameters, 1) == (1:obj.numParameters)');

% Create the new chromosome
new_chromosome = recombinationMask.*(agents(1).chromosome + obj.F*(agents(2).chromosome-agents(3).chromosome)) + ...
                (1-recombinationMask).*obj.chromosome;
            
% Penalty on violation of boundaries
violation_lb = new_chromosome<obj.lb;
violation_Ub = new_chromosome>obj.Ub;
new_penalty = sum(-(new_chromosome(violation_lb) - obj.lb(violation_lb))) + ...
              sum(+(new_chromosome(violation_Ub) - obj.Ub(violation_Ub)));

% Update Fitness
new_fitness = p_Fitness(new_chromosome, p_fitnessOptions) + new_penalty;

% Update crhomosome
if new_fitness <= obj.fitness
    obj.chromosome = new_chromosome;
    obj.fitness = new_fitness;
end

end

