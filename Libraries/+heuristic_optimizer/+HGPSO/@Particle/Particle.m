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

classdef Particle < matlab.mixin.Copyable
% Main class defining the Particle object inside the Particle Swarm Optimization algorithm
    
	properties
		% Individual properties
        position;
        fitness;
        velocity;
        
		% personal Best properties
		pBest;
        pBestFitness;
        
		% global Best properties
		gBest;
        neighborhood;
        subSwarmIndex;
		
		% Boundaries
        lb;
        Ub;
        penalty;
		
		% Flags
        binaryFlag;
        fullyInformedFlag;
        guaranteedFlag;
        
		% Maximum allowed variation
		guaranteedDelta;	% maximum variation in the guaranteed convergence PSO 
        maxVelocity;        % maximum variation of the velocity
			
		% Number of parameters
        numParameters;
		
		% Threshold used for identify if the position is close to the global best in order to apply
		% the guaranteed convergence PSO
        stopThreshold;
			
		% PSO parameters
        w;
        c_p;
        c_g;
    end
    
    methods
        function obj = Particle(p_numParameters, p_generalOptions, p_PSOOptions)
			% Particle: Constructor of the Particle object
			%
			% Input:
			% p_numParameters: number of parameters to optimize
			% p_generalOptions: General options of the PSO algorithm
			% p_PSOOptions: PSO options containing the configuration of w, c_p and c_g
			%
			% Output:
			% obj: Particle class object
			
			% Store number of parameters
            obj.numParameters = p_numParameters;
            
			% Store boundaries
			obj.lb = p_generalOptions.lb;
            obj.Ub = p_generalOptions.Ub;
        
			% Store flags
            obj.binaryFlag = p_generalOptions.binaryFlag;
            obj.fullyInformedFlag = p_generalOptions.fullyInformedFlag;
            obj.guaranteedFlag = p_generalOptions.guaranteedFlag;
            
			% Store maximum allowed variations
			obj.guaranteedDelta = p_generalOptions.guaranteedDelta;
            obj.maxVelocity = p_generalOptions.maxVelocity*ones(obj.numParameters, 1);
			
			% Store stop threshold value 
            obj.stopThreshold = p_generalOptions.stopThreshold;
            
			% Store PSO parameters
            obj.w = p_PSOOptions.w;
            obj.c_p = p_PSOOptions.c_p;
            obj.c_g = p_PSOOptions.c_g;
        end
    end
end