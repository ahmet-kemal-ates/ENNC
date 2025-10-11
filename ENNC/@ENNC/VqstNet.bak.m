% Quasi-Stationary Network
function Vqst = VqstNet(obj, u)
  Nin = length(u);
  expandedInput = zeros(1, length(obj.Vqst_w.W_h2o));
  counter = 1;
  for n=0:obj.num_cheby
      expandedInput(counter:counter+Nin-1) = cheby(u, n);
      counter=counter+Nin;
  end
  for n=1:obj.num_trig
      expandedInput(counter:counter+Nin-1) = sin(n*u);
      expandedInput(counter+Nin:counter+2*Nin-1) = cos(n*u);
      counter=counter+2*Nin;
  end 
  x = expandedInput*obj.Vqst_w.W_h2o{1} + obj.Vqst_w.W_h2o{2};
  Vqst = sigmoid(x);
end

function y = cheby(x, n)
mask = abs(x)<=0;
y = cos(n*acos(x)).*mask + (1/2.*((x-sqrt(x.^2-1)).^n + (x+sqrt(x.^2-1)).^n)).*(1-mask);
end

function y = sigmoid(x)
y = 1./(1+exp(-x));
end

