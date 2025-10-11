%%% UKF script

close all;
clear;
clc;

% Add path
addpath('../../Libraries');
addpath('../');

% Import Libriries.
import('kalmanfilter.SRUKF');

%% Configure Test
% Initial SoC
s0 = 0.5;

% Temporal offset in hours
offset = 1;

%% Load model parameters
[testName, testPath] = uigetfile('Model','Select Model', '../System Identification');
load(strcat(testPath,testName));

%% Load Test Set
[testName, testPath] = uigetfile('Test set','Select Test Set', '../../../Dataset');
load(strcat(testPath,testName));

% Check sampling time
if ~exist('Ts', 'var')
    warning('Sampling time ''Ts'' is not present in the selected dataset. The sampling time of the model will be used');
    Ts = netWeights.Ts;
    tt = Time(1):Ts:Time(end);
    Iin = interp1(Time, Iin, tt)';
    Vout = interp1(Time, Vout, tt)';
    if exist('Temp', 'var')
        Temp = interp1(Time, Temp, tt)';
    end
    SoC = interp1(Time, SoC, tt)';
    Time = tt';
elseif Ts ~= netWeights.Ts
    warning('The sampling time ''Ts'' of test set is different from that of the model. The sampling time of the model will be used');
    Ts = netWeights.Ts;
    tt = Time(1):Ts:Time(end);
    Iin = interp1(Time, Iin, tt)';
    Vout = interp1(Time, Vout, tt)';
    if exist('Temp', 'var')
        Temp = interp1(Time, Temp, tt)';
    end
    SoC = interp1(Time, SoC, tt)';
    Time = tt';
end

% Setup starting point
startIndex = 1 + round(offset*3600/Ts);

Iin_n = Iin/netWeights.maxInputITr;
Vout_n = (Vout - netWeights.minInputVTr)/(netWeights.maxInputVTr - netWeights.minInputVTr);
if  exist('Temp', 'var')
    Temp_n = Temp/netWeights.maxInputTTr;
end

%% Instantiate Model
cell = ENNC(netWeights);

%% Configure SRUKF
signalLength = length(Iin);
numTau = length(netWeights.tauDyn_w.gain);

% Set initial mean state
X0 = [s0; zeros(numTau, 1)];

% State dimension
N = length(X0);
if cell.tempFlag
    numInputs = 2;
else
    numInputs = 1;
end
numOutputs = 1;

% Set initial mean covariance   
Psoc = 0.1;
Pdyn = diag(1e-4*ones(numTau,1));
P0 = blkdiag(Psoc, Pdyn);

% Set error covariance matricies
Qsoc = 1e-10;
Qdyn = diag(1e-12*ones(numTau,1));
Q = chol(blkdiag(Qsoc,Qdyn));
R = sqrt(1e-5);

% Set UKF parameters
alpha = 0.5;
beta = 2;
kappa = 0;

X = zeros(signalLength, N);
Y = zeros(signalLength, 1);

X_n = zeros(signalLength, N);
Y_n = zeros(signalLength, 1);

modelOptions = [];

lb = [0; -inf(numTau,1)];
Ub = [1.02;  inf(numTau,1)];

ProccessFnc = @cell.ProcessFunction;
MeasurementFnc = @cell.MeasurementFunction;

SoCEstimator = SRUKF(N, numInputs, numOutputs,  ...
        ProccessFnc, MeasurementFnc,   ...
        'lb',       lb,                                     ...
        'Ub',       Ub,                                     ...
        'X0',       X0,                                     ...
        'P0',       P0,                                     ...
        'Q',        Q,                                      ...
        'R',        R,                                      ...
        'alpha',    alpha,                                  ...
        'beta',     beta,                                   ...
        'kappa',    kappa);
    
X(1,:) = SoCEstimator.X;
Y(1) = SoCEstimator.Y;

% Start UKF
if cell.tempFlag
    input = [Iin_n, Temp_n];
else
    input = Iin_n;
end
figure(1);
plt = plot((1:1)/3600, X(1:1,1)*100, (1:1)/3600, SoC(1:1,1)*100); 
iterTime = zeros(signalLength,1);
for k = 2:signalLength
    if k>startIndex
        timer = tic;
        SoCEstimator = SoCEstimator.FilterStep(input(k,:), Vout_n(k)); 
        iterTime(k) = toc(timer);
    end
    X(k,:) = SoCEstimator.X;
    Y(k) = SoCEstimator.Y;
    
%     if mod(k,3600) == 0
%         set(plt(1), 'YData', X(1:k,1)*100);
%         set(plt(1), 'XData',  (1:k)/3600);
%         set(plt(2), 'YData', SoC(1:k,1)*100);
%         set(plt(2), 'XData',  (1:k)/3600);
%         drawnow;
%     end
end

SoC_estimated = X(:,1);
plot(Time/3600, SoC, Time(startIndex:end)/3600, SoC_estimated(startIndex:end)); 
xlabel('Time [h]');
ylabel('SoC [%]');
SoCmse = immse(SoC_estimated(startIndex:end),SoC(startIndex:end));

figure
Vout_estimated = Y;
plot(Time/3600, Vout, ...
    Time(startIndex:end)/3600, minInputVTr + Vout_estimated(startIndex:end)*(maxInputVTr-minInputVTr)); 
xlabel('Time [h]');
ylabel('Voltage [V]');

avgTime = mean(iterTime(startIndex:end));

fprintf('SoC MSE: %.2e - Avg Iter Time: %.2e s\n', SoCmse, avgTime);
