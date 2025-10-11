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

classdef SRUKF
   properties(SetAccess = public)
       % State
       X;
       S;
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
       
       % UKF parameters
       lambda;
       
       % UKF weights
       wm;
       wc;
       
       % Process and measurement functions
       F;
       G;
       
       % Adaptive UKF
       residual;
       windowLength;
   end
   
   methods
       % Constructor
       function obj = SRUKF(p_numStates, p_numInputs, p_numOutputs, p_F, p_G, varargin)
           % Unscented Kalman Filter
           % Mandatory Inputs:
           %    - p_numStates: Number of the state variables.
           %    - p_numInputs: Number of the input variables.
           %    - p_numOutputs: Number of the output variables.
           %    - p_F: Handle to the process function of the system.
           %    - p_G: Handle to the measurement function of the system.
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
           %    - p_alpha: Alpha coefficient of UKF. 
           %               Default value: 1e-1.
           %    - p_beta: Beta coefficient of UKF.
           %              Default value: 2.
           %    - p_kappa: Kappa coefficient of UKF.
           %               Default value: 0.
           % Output: 
           %    - obj: Class variable
           
           if nargin < 5
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
           
           % Default UKF parameters
           default_alpha = 0.5;
           default_beta = 2;
           default_kappa = 0;
           
           default_windowLength = 60;
           
           % Define the input parser
           p = inputParser;
           p.addRequired('p_numStates');
           p.addRequired('p_numInputs');
           p.addRequired('p_numOutputs');
           p.addRequired('p_F');
           p.addRequired('p_G');
           p.addParameter('lb', default_lb);
           p.addParameter('Ub', default_Ub);
           p.addParameter('modelOptions', default_modelOptions);
           p.addParameter('X0', default_X0);
           p.addParameter('P0', default_P0);
           p.addParameter('Q', default_Q);
           p.addParameter('R', default_R);
           p.addParameter('alpha', default_alpha);
           p.addParameter('beta', default_beta);
           p.addParameter('kappa', default_kappa);
           p.addParameter('windowLength', default_windowLength);
           
           % Parse arguments
           p.parse(p_numStates, p_numInputs, p_numOutputs, p_F, p_G, varargin{:});
            
           % State and covariange matrix
           obj.X = p.Results.X0;
           obj.S = chol(p.Results.P0, 'upper');
           
           % Noise covariances
           obj.Q = p.Results.Q;
           obj.R = p.Results.R;
           
           % Dimensions
           obj.N = p_numStates;
           obj.numInputs = p_numInputs;
           obj.numOutputs = p_numOutputs;
           obj.sigmaLength = 2*obj.N+1;
           
           % UKF parameters
           alpha = p.Results.alpha;
           beta = p.Results.beta;
           kappa = p.Results.kappa;
           
           % UKF lambda coefficient
           obj.lambda = alpha^2*(obj.N + kappa) - obj.N;
 
           % Allocate memory for weights
           obj.wm = zeros(obj.sigmaLength, 1);
           obj.wc = zeros(obj.sigmaLength, 1);
           
           % Weights for mean state
           obj.wm(1) = obj.lambda /(obj.lambda + obj.N);
           obj.wm(2:obj.sigmaLength) = 1 / (2*(obj.lambda + obj.N));
           
           % Weights for covariance matrix  
           obj.wc(1) = (obj.lambda/(obj.lambda + obj.N)) + (1 - alpha^2 + beta);
           obj.wc(2:obj.sigmaLength) = 1 / (2*(obj.lambda + obj.N));
           
           % System equations
           obj.F = p_F;
           obj.G = p_G;
           
           % state bounds
           obj.lb = p.Results.lb;
           obj.Ub = p.Results.Ub;
           
           % System options
           obj.modelOptions = p.Results.modelOptions;
           
           % Output variable
           obj.Y = obj.G(obj.X, zeros(p_numInputs, 1), obj.modelOptions);
           
           % Residual for adaptive noise
           obj.residual = [];
           obj.windowLength = p.Results.windowLength;
       end
   end  
end