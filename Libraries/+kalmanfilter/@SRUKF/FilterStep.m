% SRUKF Step
function obj = FilterStep(obj, u, y)
   % Step of the SRUKF
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


   % Project on boundaries
   [i,~] = find(updatedSigmaX < obj.lb);
   updatedSigmaX(updatedSigmaX < obj.lb) = obj.lb(i);
   [i,~] = find(updatedSigmaX > obj.Ub);
   updatedSigmaX(updatedSigmaX > obj.Ub) = obj.Ub(i);

%           updatedSigmaX = updatedSigmaX - W^-1*D'*(D*W^-1*D')^-1*(D*updatedSigmaX - d);
   % Evaluate average state
   preX = zeros(obj.N, 1);
   for s=1:obj.sigmaLength
       preX = preX + obj.wm(s).*updatedSigmaX(:,s);
   end

   % Project on boundaries
   preX(preX < obj.lb) = obj.lb(preX < obj.lb);
   preX(preX > obj.Ub) = obj.Ub(preX > obj.Ub);

   % Evaluate average covariance matrix
   [~,preS] = qr([sqrt(obj.wc(2)).*(updatedSigmaX(:,2:end)-preX*ones(1,obj.sigmaLength-1)), obj.Q]');
   preS = preS(1:obj.N,1:obj.N);

   cholVector = sqrt(abs(obj.wc(1)))*(updatedSigmaX(:,1)-preX);
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
   obj.Py = obj.Py(obj.numOutputs,obj.numOutputs);

   cholVector = sqrt(abs(obj.wc(1)))*(sigmaY(:,1)-preY);
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
   postX = preX + K*(y - preY);
   % Project on boundaries
   postX(postX < obj.lb) = obj.lb(postX < obj.lb);
   postX(postX > obj.Ub) = obj.Ub(postX > obj.Ub);

   obj.X = postX; 
   % Update covariance
   U = K*obj.Py;
   [obj.S, ~] = cholupdate(preS, U, '-');

   % Evaluate updated output
   obj.Y = obj.G(obj.X, u, obj.modelOptions);
end