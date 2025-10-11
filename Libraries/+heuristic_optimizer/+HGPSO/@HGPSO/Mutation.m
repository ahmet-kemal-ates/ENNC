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

function mutedIndividual = Mutation(obj, individual2Mutate)
% Mutation: Apply mutation operator to the input individuals
% Gaussian noise mutation. Add a gaussian variation or commutate bits to a feature with a probability given by 1-mutationTh
%
% Input:
% obj: class object
% individual2Mutate: ParticleClass object referring to individual to mutate
%
% Output:
% mutedIndividual: ParticleClass object referring to the muted individual resulting from the mutation operator

% Extract Dimensions
numParameters = length(individual2Mutate.position);

% Get the std of mutation. 
mutationStd = obj.GAOptions.mutationStd*ones(numParameters,1);

% Initialize the new individual
mutedIndividual = copy(individual2Mutate);
mutedPosition = mutedIndividual.position;

% Create mutation mask. Were 1, that feature will be muted
mutationMask = round(rand(numParameters, 1)>obj.GAOptions.mutationTh);
	
% Mute position
% Get the width of mutation.
mutationAmount = mutationStd.*randn(numParameters, 1);

% Get the indices of binary variables
binaryFlag = obj.generalOptions.binaryFlag == 1;

% Apply binary mutation by commutating the selected bits.
mutedPosition(binaryFlag) = double(xor(individual2Mutate.position(binaryFlag), mutationMask(binaryFlag)));

% Apply random gaussian noise mutation
mutedPosition(~binaryFlag) = individual2Mutate.position(~binaryFlag) + mutationMask(~binaryFlag).*mutationStd(~binaryFlag).*mutationAmount(~binaryFlag);

% Update individual position
mutedIndividual.position = mutedPosition;

% Mute velocity
mutedIndividual.velocity = individual2Mutate.velocity + mutationMask.*mutationStd.*randn(numParameters, 1);

end

