% Rist Net
function Rist = RistNet(obj, u)
numHidden = size(obj.Rist_w.W_i2h, 1);

hidden = u;
for n=1:numHidden
    hidden = obj.hiddenActivation_Ist(hidden*obj.Rist_w.W_i2h{n,1} + obj.Rist_w.W_i2h{n,2});
end
Rist = obj.outputActivation_Ist(hidden*obj.Rist_w.W_h2o{1} + obj.Rist_w.W_h2o{2});

end
