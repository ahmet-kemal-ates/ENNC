% Rdyn Net
function Rdyn = RdynNet(obj, u)
numHidden = size(obj.Rdyn_w.W_i2h, 1);

hidden = u;
for n=1:numHidden
    hidden = obj.hiddenActivation_Dyn(hidden*obj.Rdyn_w.W_i2h{n,1} + obj.Rdyn_w.W_i2h{n,2});
end
Rdyn = obj.outputActivation_Dyn(hidden*obj.Rdyn_w.W_h2o{1} + obj.Rdyn_w.W_h2o{2});

end