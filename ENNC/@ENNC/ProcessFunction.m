% Process function
function next_x = ProcessFunction(obj, x, u, modelOptions)

if obj.tempFlag
    Iin = u(1);
    Temp = u(2);
    SoC = x(1);
    
    VdynInput = [Iin, Temp, SoC];
else
    Iin = u(1);
    SoC = x(1);
    
    VdynInput = [Iin, SoC];
end

next_x = x;

% SoC
next_x(1) = SoC + (Iin*obj.eta*obj.Ts)/(3600*obj.Cn); 

% Vdyn
Rdyn = obj.RdynNet(VdynInput);
tauDyn = obj.TauDynNet(VdynInput);
alpha = exp(-obj.Ts./tauDyn);
Vdyn = x(2:end)'.*alpha + (1-alpha).*Rdyn.*Iin;
next_x(2:end) = Vdyn';       
end