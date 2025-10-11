% tauDyn Net
function tauDyn = TauDynNet(obj, u)
numHidden = size(obj.tauDyn_w.W_i2h, 1);

hidden = u;
for n=1:numHidden
    hidden = obj.hiddenActivation_Dyn(hidden*obj.tauDyn_w.W_i2h{n,1} + obj.tauDyn_w.W_i2h{n,2});
end
tauDyn = sigmoid(hidden*obj.tauDyn_w.W_h2o{1} + obj.tauDyn_w.W_h2o{2});
tauDyn = (obj.tauDyn_w.minTau + tauDyn*(obj.tauDyn_w.maxTau-obj.tauDyn_w.minTau)).*obj.tauDyn_w.gain;
end

function y = sigmoid(x)
y = 1./(1+exp(-x));
end