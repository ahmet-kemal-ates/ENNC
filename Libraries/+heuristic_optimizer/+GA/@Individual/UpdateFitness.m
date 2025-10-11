% Update Particle
function obj = UpdateFitness(obj, p_Fitness, p_fitnessOptions)

% Update Fitness
obj.fitness = p_Fitness(obj.chromosome, p_fitnessOptions);

end 