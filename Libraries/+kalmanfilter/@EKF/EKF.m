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

classdef EKF
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
       
       % Process and measurement functions
       F;
       G;
       FJ;
       GJ;
       modelOptions;
       
       % Adaptive EKF
       residual;
   end
   
   methods
       % Constructor
       function obj = EKF(p_stateWidth, p_F, p_G, p_FJ, p_GJ, p_modelOptions, p_X0, p_P0, p_Q, p_R)
           % Extended Kalman Filter
           % Mandatory Inputs:
           %    - p_stateWidth: Width of the state space.
           %    - p_F: Handle to the process function of the system.
           %    - p_G: Handle to the measurement function of the system.
           % Optional Inputs
           %    - p_FJ: Handle to the Process Jacobian.
           %           Default value: [] -> Numerical estimation of the
           %           Jacobian
           %    - p_GJ: Handle to the Measurement Jacobian.
           %           Default value: [] -> Numerical estimation of the
           %           Jacobian
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
           
           % Handle less than required arguments
           switch nargin
               case 3
                   p_FJ = [];
                   p_GJ = [];
                   
                   p_modelOptions = [];
                                           
                   p_X0 = zeros(p_stateWidth,1);
                   p_P0 = 1e-1*eye(p_stateWidth,1);

                   p_Q = 1e-1*eye(p_stateWidth,1);
                   p_R = 1e-1*eye(p_stateWidth,1);
                   
               case 4  
                   p_GJ = [];
                   
                   p_modelOptions = [];
                   
                   p_X0 = zeros(p_stateWidth,1);
                   p_P0 = 1e-1*eye(p_stateWidth,1);

                   p_Q = 1e-1*eye(p_stateWidth,1);
                   p_R = 1e-1*eye(p_stateWidth,1);
                   
               case 5
                   p_modelOptions = [];
                   
                   p_X0 = zeros(p_stateWidth,1);
                   p_P0 = 1e-1*eye(p_stateWidth,1);

                   p_Q = 1e-1*eye(p_stateWidth,1);
                   p_R = 1e-1*eye(p_stateWidth,1);
                   
               case 6
                   p_X0 = zeros(p_stateWidth,1);
                   p_P0 = 1e-1*eye(p_stateWidth,1);
                   
                   p_Q = 1e-1*eye(p_stateWidth,1);
                   p_R = 1e-1*eye(p_stateWidth,1);
                   
               case 7
                   p_P0 = 1e-1*eye(p_stateWidth,1);
                   
                   p_Q = 1e-1*eye(p_stateWidth,1);
                   p_R = 1e-1*eye(p_stateWidth,1);
               
               case 8
                   p_Q = 1e-1*eye(p_stateWidth,1);
                   p_R = 1e-1*eye(p_stateWidth,1);
                   
               case 9
                   p_R = 1e-1*eye(p_stateWidth,1);
           end
            
           % Dimensions
           obj.N = p_stateWidth;
           
           % System equations
           obj.F = p_F;
           obj.G = p_G;
           obj.FJ = p_FJ;
           obj.GJ = p_GJ;
           
           % System options
           obj.modelOptions = p_modelOptions;
           
           % State and covariange matrix
           obj.X = p_X0;
           obj.P = p_P0;
           
           % Noise covariances
           obj.Q = p_Q;
           obj.R = p_R;
           
           % Residual for adaptive noise
           obj.residual = [];
       end
       
       % UKF Step
       function obj = EKFstep(obj, u, y)
           % Step of the EKF
           % Input:
           %    - obj: reference to class.
           %    - u: current input.
           %    - y: current output.
           % Output:
           %    - obj: reference to class.
           
           A = obj.FJ(obj.X, u, obj.modelOptions);
           
           % Predict state
           preX = obj.F(obj.X, u, obj.modelOptions);

           % Predict covariance matrix
           preP = A*obj.P*A' + obj.Q;

           %% Update
           % Evaluate output jacobian
           C = obj.GJ(obj.X, u, obj.modelOptions);

           % Estimate output
           preY = obj.G(preX, u, obj.modelOptions);

           % Evaluate Kalman Gain
           K = preP*C'/(C*preP*C'+V);

           % Update state
           obj.X = preX + K*(y - preY);
           
           % Update covariance
           obj.P = preP - K*C*preP; 

           % Evaluate updated output
           obj.Y = obj.G(obj.X, u, obj.modelOptions);
       end
       
       %%%% TODO %%%%
       % Adaptive UKF Step
%        function obj = AEKFstep(obj, u, y)
%            Step of the AEKF
%            Input:
%               - obj: reference to class.
%               - u: current input.
%               - y: current output.
%            Output:
%               - obj: reference to class.
%            
%            A = obj.FJ(obj.X, u, obj.modelOptions);
%            
%            Predict state
%            preX = obj.F(obj.X, u, obj.modelOptions);
% 
%            Predict covariance matrix
%            preP = A*obj.P*A' + obj.Q;
% 
%            % Update
%            Evaluate output jacobian
%            C = obj.GJ(obj.X, u, obj.modelOptions);
% 
%            Estimate output
%            preY = obj.G(preX, u, obj.modelOptions);
% 
%            Evaluate Kalman Gain
%            K = preP*C'/(C*preP*C'+V);
% 
%            Update state
%            obj.X = preX + K*(y - preY);
%            
%            Update covariance
%            obj.P = preP - K*C*preP; 
% 
%            Evaluate updated output
%            obj.Y = obj.G(obj.X, u, obj.modelOptions);
%             
%            Evaluate residuals
%            obj.residual = [obj.residual; y-obj.Y];
%            if length(obj.residual) > 15
%                obj.residual(1) = [];
%            end
%            Evaluate covariance matrix of residuals
%            obj.Presidual = 0;
%            for s = 1:obj.sigmaLength
%                obj.Presidual = obj.Presidual + obj.wc(s)*((y - sigmaY(:,s) - obj.residual(end))*(y - sigmaY(:,s) - obj.residual(end))');
%            end
%             
%            Adapt noise covariances with respect to residuals
%            D = sum(obj.residual.^2)/length(obj.residual);
%            obj.Q = obj.Pxy*D*obj.Pxy';
%            obj.R = D + obj.Presidual;
%        end
   end  
end