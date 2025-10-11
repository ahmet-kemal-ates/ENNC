function Vqst = VqstNet(obj, u)
% Quasi-Stationary Network

% Get Numer of Inputs
Nin = length(u);

% Initialize input expansion 
expandedInput = zeros(1, length(obj.Vqst_w.W_h2o{1}));

% Set first component
index = 1;

% Chebychev Expansion
n=1;
while (n <= obj.Vqst_w.num_cheby) && obj.Vqst_w.num_cheby ~= 0
    expandedInput(index:index+Nin-1) = cheby(u, n);
    index = index+Nin;
    n = n+1;
end

% Trigonometric expansion
n=1;
while n <= obj.Vqst_w.num_trig && obj.Vqst_w.num_trig ~= 0
    expandedInput(index:index+Nin-1) = sin(n * u);
    index = index+Nin;
    expandedInput(index:index+Nin-1) = cos(n * u);
    index = index+Nin;
    n = n+1;
end

% Bernstein Expansion
n = 0;
k = obj.Vqst_w.num_bernstein;
while n <= obj.Vqst_w.num_bernstein && obj.Vqst_w.num_bernstein ~= 0
    expandedInput(index:index+Nin-1) = bernstein(u, n, k, obj.binom(k+1,n+1));
    index = index+Nin;
    n = n+1;
end

% Link Functional expansion
x = expandedInput*obj.Vqst_w.W_h2o{1} + obj.Vqst_w.W_h2o{2};
Vqst = obj.outputActivation_Qst(x);
end

function y = cheby(x, n)
% First type Chebychev polynomial
mask = abs(x)<=1;
y = cos(n*acos(x)).*mask + (1/2.*((x-sqrt(x.^2-1)).^n + (x+sqrt(x.^2-1)).^n)).*(1-mask);
end

function y = bernstein(x, n, k, binomial)
% Bernstein Polynomial
% if (n >= k-n)
%    binomial= prod(n+1:k) / prod(2:k-n);
% else
%    binomial= prod(k-n+1:k) / prod(2:n);
% end
%binomial = factorial(k)/(factorial(n)*factorial(k-n));
%binomial = nchoosek(k,n);
%binomial = obj.binom(k+1,n+1);
y = binomial .* x.^n .* (1-x).^(k-n);
end
        

