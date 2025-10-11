% Adaptive SRUKF Step
function obj = AdaptiveFilterStep(obj, u, y)
   % Step of the Adaptive SRUKF
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
   sigmaMatrix = sqrt(obj.N+obj.lambda)*obj.S;

   % Evaluate sigma points
   preSigmaX(:,1) = obj.X;
   for s = 1:obj.N
       preSigmaX(:,s+1)       = obj.X + sigmaMatrix(:, s);
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
   [~,preS] = qr([sqrt(obj.wc(2)).*(updatedSigmaX(:,2:end)-preX*ones(1,obj.sigmaLength-1)), obj.Q]');
   preS = preS(1:obj.N,1:obj.N);

   cholVector = (sqrt(abs(obj.wc(1))))*(updatedSigmaX(:,1)-preX);
   if sign(obj.wc(1))>=0
       preS = cholupdate(preS, cholVector, '+');
   else
       preS = cholupdate(preS, cholVector, '-');
   end

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
   [~,obj.Py] = qr([sqrt(obj.wc(2))*(sigmaY(2:end)-preY), obj.R]');
   obj.Py = obj.Py(1,1);

   cholVector = (sqrt(abs(obj.wc(1))))*(sigmaY(:,1)-preY);
   if sign(obj.wc(1))>=0
       obj.Py = cholupdate(obj.Py, cholVector, '+');
   else
       obj.Py = cholupdate(obj.Py, cholVector, '-');
   end

   % Evaluate state-output covariance
   obj.Pxy = zeros(obj.N, 1);
   for s= 1:obj.sigmaLength
       obj.Pxy = obj.Pxy + obj.wc(s)*((updatedSigmaX(:,s) - preX)*(sigmaY(:,s) - preY)');
   end

   %% Evaluate Kalman Gain
   K = (obj.Pxy/obj.Py')/(obj.Py);

   % Update state
   obj.X = preX + K*(y - preY);

   % Update covariance
   U = K*obj.Py;
   [obj.S] = cholupdate(preS, U, '-');

   % Evaluate updated output
   obj.Y = obj.G(obj.X, u, obj.modelOptions);

   % Evaluate residuals
   obj.residual = [obj.residual; y-obj.Y];
   if length(obj.residual) > obj.windowLength
       obj.residual(1) = [];
   end
   % Evaluate covariance matrix of residuals
   obj.Presidual = 0;
   for s = 1:obj.sigmaLength
       obj.Presidual = obj.Presidual + obj.wc(s)*((sigmaY(:,s) - y + obj.residual(end))*(sigmaY(:,s) - y + obj.residual(end))');
   end

   % Adapt noise covariances with respect to residuals
   D = sum(obj.residual.^2)/length(obj.residual);
   obj.Q = chol(K*D*K'+1e-12*eye(obj.N));
   obj.R = chol(D + obj.Presidual);
end