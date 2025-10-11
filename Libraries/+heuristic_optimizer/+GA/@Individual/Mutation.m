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

function obj = Mutation(obj)
% Mutation: Apply mutation operator to the input individuals
% Gaussian noise mutation. Add a gaussian variation or commutate bits to a feature with a probability given by 1-mutationTh
%
% Input:
% obj: Individual class
%
% Output:
% obj: Indivudal class referring to the muted individual

% Get the std of mutation.
mutationStd = obj.mutationStd*ones(obj.numParameters,1);

% Initialize the new individual
mutedChromosome = obj.chromosome;

% Create mutation mask. Were 1, that feature will be muted
mutationMask = round(rand(obj.numParameters, 1)>obj.mutationTh);

% Mute position
% Get the width of mutation.
mutationAmount = mutationStd.*randn(obj.numParameters, 1);

% Get the indices of binary variables
binaryFlag = obj.binaryFlag == 1;

% Apply binary mutation by commutating the selected bits.
mutedChromosome(binaryFlag) = double(xor(obj.chromosome(binaryFlag), mutationMask(binaryFlag)));

% Apply random gaussian noise mutation
mutedChromosome(~binaryFlag) = obj.chromosome(~binaryFlag) + mutationMask(~binaryFlag).*mutationStd(~binaryFlag).*mutationAmount(~binaryFlag);

% Update individual position
obj.chromosome = mutedChromosome;

end

