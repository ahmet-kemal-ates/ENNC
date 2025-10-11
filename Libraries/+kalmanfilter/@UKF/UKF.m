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

classdef UKF
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
   end
   
   properties(SetAccess = private, Hidden = true)
       % Dimensions
       N;
       sigmaLength;
       
       % UKF parameters
       lambda;
       
       % UKF weights
       wm;
       wc;
       
       % Process and measurement functions
       F;
       G;
       modelOptions;
       
       % Adaptive UKF
       residual;
   end
   
   methods
       % Constructor
       function obj = UKF(p_stateWidth, p_F, p_G, p_modelOptions, p_X0, p_P0, p_Q, p_R, p_alpha, p_beta, p_kappa)
           % Unscented Kalman Filter
           % Mandatory Inputs:
           %    - p_stateWidth: Width of the state space.
           %    - p_F: Handle to the process function of the system.
           %    - p_G: Handle to the measurement function of the system.
           % Optional Inputs
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
           
           % Handle less than required arguments
           switch nargin
               case 3
                   p_modelOptions = [];
                                           
                   p_X0 = zeros(p_stateWidth,1);
                   p_P0 = 1e-1*eye(p_stateWidth,1);

                   p_Q = 1e-1*eye(p_stateWidth,1);
                   p_R = 1e-1*eye(p_stateWidth,1);

                   p_alpha = 1e-1;
                   p_beta = 2;
                   p_kappa = 0;
                   
               case 4                   
                   p_X0 = zeros(p_stateWidth,1);
                   p_P0 = 1e-1*eye(p_stateWidth,1);

                   p_Q = 1e-1*eye(p_stateWidth,1);
                   p_R = 1e-1*eye(p_stateWidth,1);

                   p_alpha = 1e-1;
                   p_beta = 2;
                   p_kappa = 0;
                   
               case 5
                   p_P0 = 1e-1*eye(p_stateWidth,1);

                   p_Q = 1e-1*eye(p_stateWidth,1);
                   p_R = 1e-1*eye(p_stateWidth,1);

                   p_alpha = 1e-1;
                   p_beta = 2;
                   p_kappa = 0;
                   
               case 6
                   p_Q = 1e-1*eye(p_stateWidth,1);
                   p_R = 1e-1*eye(p_stateWidth,1);

                   p_alpha = 1e-1;
                   p_beta = 2;
                   p_kappa = 0;
                   
               case 7
                   p_R = 1e-1*eye(p_stateWidth,1);

                   p_alpha = 1e-1;
                   p_beta = 2;
                   p_kappa = 0;
                   
               case 8
                   p_alpha = 1e-1;
                   p_beta = 2;
                   p_kappa = 0;
                   
               case 9
                   p_beta = 2;
                   p_kappa = 0;
                   
               case 10
                   p_kappa = 0;
           end
           
           % State and covariange matrix
           obj.X = p_X0;
           obj.P = p_P0;
           
           % Noise covariances
           obj.Q = p_Q;
           obj.R = p_R;
           
           % Dimensions
           obj.N = p_stateWidth;
           obj.sigmaLength = 2*obj.N+1;
           
           % UKF lambda coefficient
           obj.lambda = p_alpha^2*(obj.N + p_kappa) - obj.N;
 
           % Allocate memory for weights
           obj.wm = zeros(obj.sigmaLength, 1);
           obj.wc = zeros(obj.sigmaLength, 1);
           
           % Weights for mean state
           obj.wm(1) = obj.lambda /(obj.lambda + obj.N);
           obj.wm(2:obj.sigmaLength) = 1 / (2*(obj.lambda + obj.N));
           
           % Weights for covariance matrix  
           obj.wc(1) = (obj.lambda/(obj.lambda + obj.N)) + (1 - p_alpha^2 + p_beta);
           obj.wc(2:obj.sigmaLength) = 1 / (2*(obj.lambda + obj.N));
           
           % System equations
           obj.F = p_F;
           obj.G = p_G;
           
           % System options
           obj.modelOptions = p_modelOptions;
           
           % Residual for adaptive noise
           obj.residual = [];
       end
       
       % UKF Step
       function obj = FilterStep(obj, u, y)
           % Step of the UKF
           % Input:
           %    - obj: reference to class.
           %    - u: current input.
           %    - y: current output.
           % Output:
           %    - obj: reference to class.
           
           % Allocate variables for sigma points
           preSigmaX = zeros(obj.N, obj.sigmaLength);
           updatedSigmaX = zeros(obj.N, obj.sigmaLength);
           sigmaY = zeros(1, obj.sigmaLength);
           
           % Evaluate the sigma matrix
           sigmaMatrix = chol((obj.N+obj.lambda)*obj.P, 'upper');

           % Evaluate sigma points
           preSigmaX(:,1) = obj.X;
           for s = 1:obj.N
               preSigmaX(:,s+1)   = obj.X + sigmaMatrix(:, s);
               preSigmaX(:,obj.N+s+1) = obj.X - sigmaMatrix(:, s);
           end
     
           % Update sigma points
           for s = 1:obj.sigmaLength
               updatedSigmaX(:,s) = obj.F(preSigmaX(:,s), u, obj.modelOptions);
           end
     
           % Evaluate average state
           preX = zeros(obj.N, 1);
           for s=1:obj.sigmaLength
               preX = preX + obj.wm(s).*updatedSigmaX(:,s);
           end
            
           % Evaluate average covariance matrix
           preP = zeros(obj.N, obj.N);
           for s=1:obj.sigmaLength
               preP = preP + obj.wc(s).*((updatedSigmaX(:,s) - preX)*(updatedSigmaX(:,s) - preX)');
           end
           preP = preP + obj.Q;
     
           %% Update
           % Evaluate output for each new sigma point
           for s=1:obj.sigmaLength
               sigmaY(:,s) = obj.G(updatedSigmaX(:,s), u, obj.modelOptions);
           end

           % Evaluate mean estimated output
           preY = 0;
           for s=1:obj.sigmaLength
               preY = preY + obj.wm(s)*sigmaY(:,s);
           end

           % Evaluate covariance output
           obj.Py = 0;
           for s=1:obj.sigmaLength
               obj.Py = obj.Py + obj.wc(s) * ((sigmaY(:,s) - preY)*(sigmaY(:,s) - preY)');
           end
           obj.Py = obj.Py + obj.R;

           % Evaluate state-output covariance
           obj.Pxy = zeros(obj.N, 1);
           for s= 1:obj.sigmaLength
               obj.Pxy = obj.Pxy + obj.wc(s)*((updatedSigmaX(:,s) - preX)*(sigmaY(:,s) - preY)');
           end
     
           %% Evaluate Kalman Gain
           K = obj.Pxy * (obj.Py^-1);

           % Update state
           obj.X = preX + K*(y - preY);
            
           % Update covariance
           obj.P = preP - K*obj.Py*K';
            
           % Evaluate updated output
           obj.Y = obj.G(obj.X, u, obj.modelOptions);
       end
       
       % Adaptive UKF Step
       function obj = AdaptiveFilterStep(obj, u, y)
           % Step of the AUKF
           % Input:
           %    - obj: reference to class.
           %    - u: current input.
           %    - y: current output.
           % Output:
           %    - obj: reference to class.
           
           % Allocate variables for sigma points
           preSigmaX = zeros(obj.N, obj.sigmaLength);
           updatedSigmaX = zeros(obj.N, obj.sigmaLength);
           sigmaY = zeros(1, obj.sigmaLength);
           
           % Evaluate the sigma matrix
           sigmaMatrix = chol((obj.N+obj.lambda)*obj.P, 'upper');

           % Evaluate sigma points
           preSigmaX(:,1) = obj.X;
           for s = 1:obj.N
               preSigmaX(:,s+1) = obj.X + sigmaMatrix(:, s);
               preSigmaX(:,obj.N+s+1) = obj.X - sigmaMatrix(:, s);
           end
     
           % Update sigma points
           for s = 1:obj.sigmaLength
               updatedSigmaX(:,s) = obj.F(preSigmaX(:,s), u, obj.modelOptions);
           end
     
           % Evaluate average state
           preX = zeros(obj.N, 1);
           for s=1:obj.sigmaLength
               preX = preX + obj.wm(s).*updatedSigmaX(:,s);
           end
            
           % Evaluate average covariance matrix
           preP = zeros(obj.N, obj.N);
           for s=1:obj.sigmaLength
               preP = preP + obj.wc(s).*((updatedSigmaX(:,s) - preX)*(updatedSigmaX(:,s) - preX)');
           end
           preP = preP + obj.Q;
     
           %% Update
           % Evaluate output for each new sigma point
           for s=1:obj.sigmaLength
               sigmaY(:,s) = obj.G(updatedSigmaX(:,s), u, obj.modelOptions);
           end

           % Evaluate mean estimated output
           preY = 0;
           for s=1:obj.sigmaLength
               preY = preY + obj.wm(s)*sigmaY(:,s);
           end

           % Evaluate covariance output
           obj.Py = 0;
           for s=1:obj.sigmaLength
               obj.Py = obj.Py + obj.wc(s) * ((sigmaY(:,s) - preY)*(sigmaY(:,s) - preY)');
           end
           obj.Py = obj.Py + obj.R;

           % Evaluate state-output covariance
           obj.Pxy = zeros(obj.N, 1);
           for s= 1:obj.sigmaLength
               obj.Pxy = obj.Pxy + obj.wc(s)*((updatedSigmaX(:,s) - preX)*(sigmaY(:,s) - preY)');
           end
     
           %% Evaluate Kalman Gain
           K = obj.Pxy * (obj.Py^-1);

           % Update state
           obj.X = preX + K*(y - preY);
            
           % Update covariance
           obj.P = preP - K*obj.Py*K';
            
           % Evaluate updated output
           obj.Y = obj.G(obj.X, u, obj.modelOptions);
           
           % Evaluate residuals
           obj.residual = [obj.residual; y-obj.Y];
           if length(obj.residual) > 15
               obj.residual(1) = [];
           end
           % Evaluate covariance matrix of residuals
           obj.Presidual = 0;
           for s = 1:obj.sigmaLength
               obj.Presidual = obj.Presidual + obj.wc(s)*((y - sigmaY(:,s) - obj.residual(end))*(y - sigmaY(:,s) - obj.residual(end))');
           end
            
           % Adapt noise covariances with respect to residuals
           D = sum(obj.residual.^2)/length(obj.residual);
           obj.Q = K*D*K';
           obj.R = D + obj.Presidual;
       end
   end  
end