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

function obj = UpdateSwarm(obj)
% UpdateGBest: Update the globl bests positions and the particle subswarm memberships
%
% Input:
% obj: HGPSO class object
%
% Output:
% obj: HGPSO class object

% Import Library
import heuristic_optimizer.HGPSO.*

% Get the swarm and the related fitness function and fitness options
localSwarm = obj.swarm;
localFitness = obj.Fitness;
localFitnessOptions = obj.fitnessOptions;
% Update the swarm
for p=1:obj.generalOptions.numIndividuals
    neighborhood = [localSwarm.subSwarmIndex] == localSwarm(p).subSwarmIndex;
    localSwarm(p).neighborhood = [localSwarm(neighborhood).pBest];
end
parforArg = feature('numcores')*obj.generalOptions.parallelFlag;
parfor (p=1:obj.generalOptions.numIndividuals, parforArg)
    localSwarm(p) = localSwarm(p).UpdateParticle(localFitness, localFitnessOptions);
end
obj.swarm = localSwarm;

end