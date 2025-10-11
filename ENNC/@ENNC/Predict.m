function [Vout, Vist, Rist, Vdyn, Rdyn, tauDyn, Vqst] = Predict(obj, u, X0)

numSamples = size(u,1);
numTau = length(X0)-1;

X = X0'.*ones(numSamples, length(X0));
Vout = zeros(numSamples, 1);
Vist = zeros(numSamples, 1);
Vdyn = zeros(numSamples, numTau);
Vqst = zeros(numSamples, 1);
Rist = zeros(numSamples, 1);
Rdyn = zeros(numSamples, numTau);
tauDyn = zeros(numSamples, numTau);

for k=1:numSamples-1
    X(k+1, :) = obj.ProcessFunction(X(k,:)', u(k,:));
    Vout(k) = obj.MeasurementFunction(X(k,:)', u(k,:));
    Vdyn(k,:) = X(k,2:end);
    
    if obj.tempFlag
        SoC = X(k,1);
        Iin = u(k,1);
        Temp = u(k,2);
        RistInput = [Iin, Temp, SoC];
        VdynInput = [Iin, Temp, SoC];
        VqstInput = [SoC, Temp];
    else  
        SoC = X(k,1);
        Iin = u(k,1);
        RistInput = [Iin, SoC];
        VdynInput = [Iin, SoC];
        VqstInput = SoC;
    end
    Rist(k) = obj.RistNet(RistInput);
    Rdyn(k,:) = obj.RdynNet(VdynInput);
    tauDyn(k,:) = obj.TauDynNet(VdynInput);
    
    Vist(k) = Rist(k)*Iin;
    Vqst(k) = obj.VqstNet(VqstInput);
end

end


