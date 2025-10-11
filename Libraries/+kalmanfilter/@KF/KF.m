% Copyright (c) 2017 Massimiliano Luzi
% University of Rome "La Sapienza"
% 
% massimiliano.luzi@uniroma1.it 
% 
% This file is part of "Kalman Library".
% 
% "Kalman Library" is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% "Kalman Library" is distributed in the hope that it will be useful. 
% IT IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
% INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
% PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
% HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
% CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE 
% OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 
% See the GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with "Kalman Library".If not, see<http://www.gnu.org/licenses/>.

classdef KF
   properties(SetAccess = public)
       % State
       X;
       P;
       Y;
       % Noise covariance
       Q;
       R;
       % Covariances
       Py;
       Pxy;
       Presidual;
       % Boundaries
       lb;
       Ub;
       % modelOptions
       modelOptions;
   end
   
   properties(SetAccess = public, Hidden = true)
       % Dimensions
       N;
       numInputs;
       numOutputs;
       sigmaLength;
              
       % Process and measurement functions
       A;
       B;
       C;
       D;
       
       % Adaptive KF
       residual;
   end
   
   methods
       % Constructor
       function obj = KF(p_numStates, p_numInputs, p_numOutputs, p_A, p_B, p_C, p_D, varargin)
           % Kalman Filter
           % Mandatory Inputs:
           %    - p_numStates: Number of the state variables.
           %    - p_numInputs: Number of the input variables.
           %    - p_numOutputs: Number of the output variables.
           %    - p_A: Process matrix.
           %    - p_B: Process input matrix.
           %    - p_C: Measurement matrix.
           %    - p_D: Measurement input matrix.
           % Optional Inputs
           %    - p_lB: lower bounds for states
           %    - p_UB: upper bouond for states
           %    - p_modelOptions: struct containing optional variables to be used 
           %                      for the evaluation of f(x,u) and g(x,u).
           %                      Default value: [].
           %    - p_X0: Initial state. 
           %            Default value: zeros(p_stateWidth,1).
           %    - p_P0: Initial covariance. 
           %            Default value: 1e-1*eye(p_stateWidth,1).
           %    - p_Q:  Initial process noise covariance. 
           %            Default value: 1e-1*eye(p_stateWidth,1).
           %    - p_R:  Initial measurement noise covariance.
           %            Default value: 1e-1*eye(p_stateWidth,1).
           % Output: 
           %    - obj: Class variable
           
           if nargin < 7
               error('The constructor must receive at leas five arguments: 1. The number of state variables; 2. The number of input variables 3. The number of output variables; 4. The process function; 5. the measurement function.');
           end
            
           % Handle less than required arguments
           
           % Default bounds
           default_lb = -inf(p_numStates, 1);
           default_Ub =  inf(p_numStates, 1);
           
           % Default modelOptions
           default_modelOptions = [];
           
           % Default initial state
           default_X0 = zeros(p_numStates, 1);
           
           % Default initial covariance matrix
           default_P0 = 1e-1*eye(p_numStates);
           
           % Default process noise covariance matrices
           default_Q = 1e-4*eye(p_numStates);
           default_R = 1e-4*ones(p_numOutputs, 1);
           
           % Define the input parser
           p = inputParser;
           p.addRequired('p_numStates');
           p.addRequired('p_numInputs');
           p.addRequired('p_numOutputs');
           p.addRequired('p_A');
           p.addRequired('p_B');
           p.addRequired('p_C');
           p.addRequired('p_D');
           p.addParameter('lb', default_lb);
           p.addParameter('Ub', default_Ub);
           p.addParameter('modelOptions', default_modelOptions);
           p.addParameter('X0', default_X0);
           p.addParameter('P0', default_P0);
           p.addParameter('Q', default_Q);
           p.addParameter('R', default_R);
           
           % Parse arguments
           p.parse(p_numStates, p_numInputs, p_numOutputs, p_A, p_B, p_C, p_D, varargin{:});
            
           % State and covariange matrix
           obj.X = p.Results.X0;
           obj.P = p.Results.P0;
           
           % Noise covariances
           obj.Q = p.Results.Q;
           obj.R = p.Results.R;
           
           % Dimensions
           obj.N = p_numStates;
           obj.numInputs = p_numInputs;
           obj.numOutputs = p_numOutputs;
           
           % System matrices
           obj.A = p_A;
           obj.B = p_B;
           obj.C = p_C;
           obj.D = p_D;
           
           % state bounds
           obj.lb = p.Results.lb;
           obj.Ub = p.Results.Ub;
           
           % System options
           obj.modelOptions = p.Results.modelOptions;
           
           % Output variable
           obj.Y = obj.C*obj.X + obj.D*zeros(p_numInputs, 1);
           
           % Residual for adaptive noise
           obj.residual = [];
       end
   end  
end