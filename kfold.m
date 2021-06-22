%%Split preprocess data 5 fold

%Clear workspace
clearvars; close all; clc;

gpuDevice(2)

destinationOut = '';

%% create datastores for processed labels and images
% Images Datapath % define reader
procVolReader = @(x) niftiread(x);
procVolLoc = fullfile(destination,'imgNormalized');
procVolDs = imageDatastore(procVolLoc, ...
    'FileExtensions','.nii','LabelSource','foldernames','ReadFcn',procVolReader);

procLblReader =  @(x) uint8(niftiread(x));
procLblLoc = fullfile(destination,'lblOrient');
classNames = ["background", "tumour", "nonenhancingtumor" "enhancingtumor"];
pixelLabelID = [0 1 2 3]; 
procLblDs = pixelLabelDatastore(procLblLoc,classNames,pixelLabelID, ...
    'FileExtensions','.nii','ReadFcn',procLblReader);

%kfold partition
num_images = length(procVolDs.Labels); %number of obervations for kfold
c1 = cvpartition(num_images,'kfold',5);
err = zeros(c1.NumTestSets,1);

C = cell(1,2);
[idxTest] = deal(C);

for idxFold = 1:c1.NumTestSets
    idxTest{idxFold} = test(c1,idxFold); %logical indices for test set
    save('idxTest.mat','idxTest');

    idxHold{idxFold} = training(c1,idxFold); %logical indices for training set-holdout partition
    save('idxHold.mat','idxHold');
    imdsHold = subset(procVolDs,idxHold{idxFold}); %imds for holdout partition
    pxdsHold = subset(procLblDs,idxHold{idxFold}); %imds for holdout partition

    num_imdsHold = length(imdsHold.Labels); %number of obervations for holdout partition
    c2 = cvpartition(num_imdsHold,'holdout',0.20);

    idxVal{idxFold} = test(c2); %logical indices for val set
    save('idxVal.mat','idxVal');

    idxTrain{idxFold} = training(c2); %logical indices for training set
    save('idxTrain.mat','idxTrain');

end
