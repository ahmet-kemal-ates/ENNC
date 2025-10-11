classdef Individual < matlab.mixin.Copyable
    properties
        chromosome;
        fitness;
        penalty;
        
        F;

        binaryFlag;
        
        lb;
        Ub;
        
        crossoverTh;
        
        numParameters;
    end
    
    methods
        %% Contructor
        function obj = Individual(p_numParameters, p_generalOptions, p_DEOptions)
            obj.numParameters = p_numParameters;
          
            obj.lb = p_generalOptions.lb;
            obj.Ub = p_generalOptions.Ub;
        
            obj.binaryFlag = p_generalOptions.binaryFlag;
            
            obj.F = p_DEOptions.F;
            obj.crossoverTh = p_DEOptions.crossoverTh;
        end
    end
end