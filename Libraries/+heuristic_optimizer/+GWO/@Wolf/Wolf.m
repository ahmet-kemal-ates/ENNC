classdef Wolf
    properties
        
        position;
        fitness;
        
        % Maximum allowed variation
        maxDelta;
        
        % Boundaries
        lb;
        Ub;
        penalty;
        
        % Number of parameters
        numParameters;
    end
    
    methods
        % Contructor
        function obj = Wolf(p_numParameters, p_generalOptions)
            % Store number of parameters
            obj.numParameters = p_numParameters;
            
			% Store boundaries
			obj.lb = p_generalOptions.lb;
            obj.Ub = p_generalOptions.Ub;
            
            % Store maximum allowed variations
			obj.maxDelta = p_generalOptions.maxDelta;
        end
    end
end