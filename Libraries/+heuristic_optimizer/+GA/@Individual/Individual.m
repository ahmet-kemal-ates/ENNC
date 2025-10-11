classdef Individual < matlab.mixin.Copyable
    properties
        chromosome;
        fitness;
        penalty;
        
        elite = false;
        rank = inf;
        dominance = inf;
        
        lb;
        Ub;
        
        binaryFlag;
        mutationTh;
        mutationStd;
        crossoverTh;
        
        numParameters;
        stopThreshold;
    end
    
    methods
        %% Contructor
        function obj = Individual(p_numParameters, p_generalOptions, p_GAOptions)
            obj.numParameters = p_numParameters;
            
            obj.lb = p_generalOptions.lb;
            obj.Ub = p_generalOptions.Ub;
        
            obj.binaryFlag = p_generalOptions.binaryFlag;
            
            obj.mutationTh = p_GAOptions.mutationTh;
            obj.mutationStd = p_GAOptions.mutationStd;
            obj.crossoverTh = p_GAOptions.crossoverTh;
            
            obj.stopThreshold = p_generalOptions.stopThreshold;
        end
    end
end