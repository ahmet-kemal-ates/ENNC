% Measurement Function
function Vout = MeasurementFunction(obj, x, u, modelOptions)

% Build parameteric input
if obj.tempFlag
    Iin = u(1);
    Temp = u(2);
    SoC = x(1);
    
    VqstInput = [SoC, Temp];
    RistInput = [Iin, Temp, SoC];
else
    Iin = u(1);
    SoC = x(1);

    VqstInput = SoC;
    RistInput = [Iin, SoC];
end
          
% Vist
Rist = obj.RistNet(RistInput);
Vist = Rist*Iin;
% Vdyn
Vdyn = x(2:end)'*obj.Vdyn_w.W_mix';
% Vqst
Vqst = obj.VqstNet(VqstInput);

Vout = Vist + Vdyn + Vqst;
end