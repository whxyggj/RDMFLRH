% RDMFLRH: complete BBCSport dataset demo (Pan et al., TCE 2025).
% Run myDMF_BBCsport from this folder. Requires Statistics Toolbox.
clear;
clc;

root_dir = fileparts(mfilename('fullpath'));
addpath(root_dir, fullfile(root_dir,'third_party','tools'), ...
    fullfile(root_dir,'third_party','gspbox'));
assert(exist('knnsearch','file') == 2 && exist('randsample','file') == 2, ...
    'Statistics and Machine Learning Toolbox is required.');
rng(0,'twister');

load(fullfile(root_dir,'data','BBCSport.mat'));
X{1} = data{1}; %m*n
X{2} = data{2};
truth = double(truelabel{1}'); % n*1

class_num = length(unique(truth)); % 类别个数
maxiter = 1000;

layer_value = cell(1, 3);
layer_value{1} = [50, class_num];
layer_value{2} = [100, class_num];
layer_value{3} = [100, 50, class_num];

replic = 1; % Set to 10 for repeated training runs.

layer_idx = 1;
layers = layer_value{layer_idx};
lambda = 0.1; %0.1
gamma = 100;%100

AC_ = zeros(1, replic);
NMI_ = zeros(1, replic);
purity_ = zeros(1, replic);
Fscore_ = zeros(1, replic);
Precision_ = zeros(1, replic);
Recall_ = zeros(1, replic);
AR_ = zeros(1, replic);

res_record = zeros(7, replic); % ACC, NMI, purity, F-score, precision, recall, ARI
V_star_record = cell(1, replic);

for i = 1: replic
    tic;
    [V_star, obj, cnt] = myDMF(X, layers, 'lambda', lambda, 'gamma', gamma, 'maxiter', maxiter);
    elapsed_seconds(i) = toc;
    obj_record{i} = obj;
    error_count_record(i) = cnt;
    idx = litekmeans(V_star', class_num, 'Replicates', 20);
    labels_record{i} = idx;
    result = EvaluationMetrics(truth, idx);
    AC_(i) = result(1);
    NMI_(i) = result(2);
    purity_(i) = result(3);
    Fscore_(i) = result(4);
    Precision_(i) = result(5);
    Recall_(i) = result(6);
    AR_(i) = result(7);
    res_record(:, i) = result;
    V_star_record{i} = V_star;
end
AC(1) = mean(AC_); AC(2) = std(AC_);
NMI(1) = mean(NMI_); NMI(2) = std(NMI_);
purity(1) = mean(purity_); purity(2) = std(purity_);
Fscore(1) = mean(Fscore_); Fscore(2) = std(Fscore_);
Precision(1) = mean(Precision_); Precision(2) = std(Precision_);
Recall(1) = mean(Recall_); Recall(2) = std(Recall_);
AR(1) = mean(AR_); AR(2) = std(AR_);

assert(isequal(size(V_star),[class_num,numel(truth)]), 'Invalid representation size.');
assert(isreal(V_star) && all(isfinite(V_star(:))), 'Invalid representation.');
assert(all(isfinite(res_record(:))), 'Invalid clustering metrics.');
output_dir = fullfile(root_dir,'results');
if ~exist(output_dir,'dir'), mkdir(output_dir); end
save(fullfile(output_dir,'BBCSport_result.mat'), ...
    'res_record','V_star_record','obj_record','labels_record', ...
    'error_count_record','elapsed_seconds','layers','lambda','gamma', ...
    'maxiter','replic','AC','NMI','purity','Fscore','Precision','Recall','AR');
fprintf('BBCSport: %d samples, %d classes, %d views.\n',numel(truth),class_num,numel(X));
fprintf('ACC %.4f; NMI %.4f; ARI %.4f; iterations %d; training %.2f s\n', ...
    AC(1),NMI(1),AR(1),numel(obj_record{end}),sum(elapsed_seconds));
fprintf('Saved results to %s\n',fullfile(output_dir,'BBCSport_result.mat'));
