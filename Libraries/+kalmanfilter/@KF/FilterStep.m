% SRUKF Step
function obj = FilterStep(obj, u, y)
   % Step of the SRUKF
   % Input:
   %    - obj: reference to class.
   %    - u: current input.
   %    - y: current output.
   % Output:
   %    - obj: reference to class.

   % Predict state
   preX = obj.A*obj.X + obj.B*u;

   % Predict covariance matrix
   preP = A*obj.P*A' + obj.Q;

   %% Update
   % Estimate output
   preY = obj.C*obj.X + obj.D*u;
   
   % Evaluate Kalman Gain
   K = preP*C'/(C*preP*C'+obj.R);

   % Update state
   obj.X = preX + K*(y - preY);

   % Update covariance
   obj.P = preP - K*C*preP; 

   % Evaluate updated output
   obj.Y = obj.C*obj.X + obj.D*u;
end