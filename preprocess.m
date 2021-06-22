%% Preprocess BraTS data
%% zscore and clip at -5 to +5, check the orientation 

% Clear workspace
clearvars; close all; clc;

gpuDevice(2)

destination = '/rsrch1/ip/rmuthusivarajan/imaging/BraTS/Task01_BrainTumour/';
destinationOut = '/';

imgTr = dir(fullfile(destination, 'imagesTr','*.gz'));
imgFile = {imgTr.name}';
imgFolder = {imgTr.folder}';

lblTr = dir(fullfile(destination, 'labelsTr','*.gz'));
lblFile = {lblTr.name}';
lblFolder = {lblTr.folder}';

%create directories to store preprocessed data 
mkdir(fullfile(destinationOut,'imgNormalized'));
mkdir(fullfile(destinationOut,'lblOrient'));

for id = 1:484

%zscore images
imgLoc = fullfile(imgFolder(id),imgFile(id));
imgInfo = niftiinfo(char(imgLoc));
imgVol = niftiread(imgInfo);
            
%outV1 = single(imgVol);
chn_Mean = mean(imgVol,[1 2 3]);
chn_Std = std(imgVol,0,[1 2 3]);
scale = 1./chn_Std;
out = (imgVol - chn_Mean)./chn_Std;

rangeMin = -5;
rangeMax = 5;

out(out > rangeMax) = rangeMax;
out(out < rangeMin) = rangeMin;

normFile = ['normalized_brats_' num2str(id,'%03.f') '.nii'];
normDir = fullfile(destinationOut,'imgNormalized',normFile);
niftiwrite(out,normDir);

%Labels Orient change LPI to RAI 
lblLoc = string(fullfile(lblFolder(id),lblFile(id)));
oriFile = ['orient_brats_' num2str(id, '%03.f') '.nii'];
system(sprintf('c3d %s -orient RAI -o %s',lblLoc,oriFile));
movefile('ori*',fullfile(destinationOut,'lblOrient'));

id = id + 1;
end
