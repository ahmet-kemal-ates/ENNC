classdef ENNC
   properties(SetAccess = public, Hidden = true)
       Rist_w;
       hiddenActivation_Ist;
       outputActivation_Ist;
       
       Rdyn_w;
       tauDyn_w;
       Vdyn_w;
       hiddenActivation_Dyn;
       outputActivation_Dyn;
       
       Vqst_w
       outputActivation_Qst;
       
       Cn;
       eta;
       Ts;
       
       binom;
       
       tempFlag;
   end
   
   methods(Access = public)
       % Constructor
       function obj = ENNC(p_netWeights)
           obj.Rist_w = p_netWeights.Rist_w;
           
           switch p_netWeights.Rist_w.hiddenActivation
               case 'tanh'
                   obj.hiddenActivation_Ist = @(x) (tanh(x));
                   
               case 'relu'
                   obj.hiddenActivation_Ist = @(x) max(0, x);
           end
           
           switch p_netWeights.Rist_w.outputActivation
               case 'linear'
                   obj.outputActivation_Ist = @(x) (x);
                   
               case 'sigmoid'
                   obj.outputActivation_Ist = @(x) 1./(1+exp(-x));
           end
           
           obj.Rdyn_w = p_netWeights.Rdyn_w;
           obj.tauDyn_w = p_netWeights.tauDyn_w;
           obj.Vdyn_w = p_netWeights.Vdyn_w;
           
           switch p_netWeights.Rdyn_w.hiddenActivation
               case 'tanh'
                   obj.hiddenActivation_Dyn = @(x) (tanh(x));
                   
               case 'relu'
                   obj.hiddenActivation_Dyn = @(x) max(0, x);
           end
           
           switch p_netWeights.Rdyn_w.outputActivation
               case 'linear'
                   obj.outputActivation_Dyn = @(x) (x);
                   
               case 'sigmoid'
                   obj.outputActivation_Dyn = @(x) 1./(1+exp(-x));
           end
           
           obj.Vqst_w = p_netWeights.Vqst_w;
           switch p_netWeights.Vqst_w.outputActivation
               case 'linear'
                   obj.outputActivation_Qst = @(x) (x);
                   
               case 'sigmoid'
                   obj.outputActivation_Qst = @(x) 1./(1+exp(-x));
           end
           
           obj.Cn = p_netWeights.Cn/p_netWeights.maxInputITr;
           obj.eta = 1;
           obj.Ts = p_netWeights.Ts;
           
           obj.binom = eye(obj.Vqst_w.num_bernstein+1);
           obj.binom(2:end,:) = 0;
           obj.binom(:,1) = 1; 
           
           for n=2:obj.Vqst_w.num_bernstein+1
               for m=2:obj.Vqst_w.num_bernstein+1
                   obj.binom(n,m) = obj.binom(n-1,m-1)+obj.binom(n-1,m);
               end
           end
           
           obj.tempFlag = size(p_netWeights.Rist_w.W_i2h{1,1}, 1) == 3;
       end 
   end
end