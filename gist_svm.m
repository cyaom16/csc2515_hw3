clear all
clc

userpath_fix = userpath;
userpath_fix = userpath_fix(1:end-1);
hm_dir = [userpath_fix,filesep,'csc2515_hw',filesep];
addpath([hm_dir,'data',filesep]);
addpath([hm_dir,'gistdescriptor',filesep]);

cd(hm_dir);

%%
imds_labels = csvread('train.csv');
imds_labels = imds_labels(:,2);

imds = imageDatastore([hm_dir,filesep,'data',filesep,'train']);
imds.Labels = categorical(imds_labels);

test_set = imageDatastore([hm_dir,filesep,'data',filesep,'val']);
[train_set,validate_set] = splitEachLabel(imds,0.7,'randomize');
train_labels = train_set.Labels;

%%
tbl = countEachLabel(imds)

%%
class_1 = find(imds.Labels == '1', 1);
class_2 = find(imds.Labels == '2', 1);
class_3 = find(imds.Labels == '3', 1);
class_4 = find(imds.Labels == '4', 1);
class_5 = find(imds.Labels == '5', 1);
class_6 = find(imds.Labels == '6', 1);
class_7 = find(imds.Labels == '7', 1);
class_8 = find(imds.Labels == '8', 1);

figure
subplot(2,4,1);
imshow(readimage(imds,class_1))
subplot(2,4,2);
imshow(readimage(imds,class_2))
subplot(2,4,3);
imshow(readimage(imds,class_3))
subplot(2,4,4);
imshow(readimage(imds,class_4))
subplot(2,4,5);
imshow(readimage(imds,class_5))
subplot(2,4,6);
imshow(readimage(imds,class_6))
subplot(2,4,7);
imshow(readimage(imds,class_7))
subplot(2,4,8);
imshow(readimage(imds,class_8))

%% param setup
clear param
param.imageSize = [128,128];
param.orientationsPerScale = [8,8,8,8];
param.numberBlocks = 4;
param.fc_prefilt = 4;
n_features = sum(param.orientationsPerScale)*param.numberBlocks^2;

% % Visualization
% figure
% subplot(121)
% imshow(img1)
% title('Input image')
% subplot(122)
% showGist(gist_tmp, param)
% title('Descriptor')

%% load features
load gist_512.mat

%% load training features (load gist.mat instead)
train_features = zeros(n_features,size(train_set.Files,1));
for i = 1:size(train_features,2)
    img = imread(train_set.Files{i});
    train_features(:,i) = LMgist(img,'',param);
end

validate_features = zeros(n_features,size(validate_set.Files,1));
for i = 1:size(validate_features,2)
    img = imread(validate_set.Files{i});
    validate_features(:,i) = LMgist(img,'',param);
end

%% load classifier
t = templateSVM('KernelFunction','gaussian');
mdl = fitcecoc(train_features,train_labels,...
    'Learners',t,'Coding','onevsall','ObservationsIn','columns');

%%
rng default
Mdl = fitcecoc(train_features,train_labels,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus'))

%% cross-validated classifier using 5-fold
cv_mdl = crossval(mdl,'KFold',5);

oos_loss = kfoldLoss(cv_mdl)

%% assign predictions to validation and test sets
predicted_labels = predict(mdl,validate_features');

%test_labels = predict(classifier,test_features');

validate_labels = validate_set.Labels;
confMat = confusionmat(validate_labels, predicted_labels);

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2))

%%
table2array(tbl(:,2))'*diag(confMat)/sum(table2array(tbl(:,2)))
