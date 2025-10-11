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

function [firstSon, secondSon] = Crossover(obj, firstParent, secondParent )
% Crossover: Apply crossover operator to the input individuals
% Scattring crossover. Each features is swapped among the two parents with a certain probability given by 1-crossoverTh
%
% Input:
% obj: class object
% firstParent: Individual object referring to the first parent individual
% secondParent: Individual object referring to the second parent individual 
%
% Output:
% firstSon: Individual object referring to the first son resulting from the crossover operator
% secondSon: Individual object referring to the second son resulting from the crossover operator


% Extract dimensions
numParameters = length(firstParent.chromosome);

% Initialize son particles by copying the ParticleClass objects
firstSon = copy(firstParent);
secondSon = copy(secondParent);

% Create the mask for applying the crossover. Where crossoverMask==1 the related features will be swapped
crossoverMask = round(rand(numParameters, 1)>obj.GAOptions.crossoverTh);

% Create the first son
firstSon.chromosome = crossoverMask.*secondParent.chromosome + (1-crossoverMask).*firstParent.chromosome;

% Create the second son
secondSon.chromosome = crossoverMask.*firstParent.chromosome + (1-crossoverMask).*secondParent.chromosome;

end

