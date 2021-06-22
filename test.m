%Segmentation on Test Data

%Clear workspace
clear; close all; clc;

gpuDevice(2)

destinationOut = '';

imgDir = dir(fullfile(destination, 'imgNormalized','*.nii'));
imgFile = {imgDir.name}';
imgFolder = {imgDir.folder}';

%%Load test indices 
s = load('idxTest.mat');
c = struct2cell(s);
idxTest = cat(1,c{:});

for idxFold = 1:5
    
    disp(['Processing K-fold-' num2str(idxFold)]);

    trainedNetName = ['fold_' num2str(idxFold) '_trainedNet.mat'];
    load(fullfile(destination, trainedNetName));
          
    testSet = idxTest{1,idxFold};
    
    imgFileTest(:,idxFold) = imgFile(testSet);
    imgFolderTest(:,idxFold) = imgFolder(testSet);
   
    %create directories to store labels 
    mkdir(fullfile(destination,'predicted',['predictedLabel-fold' num2str(idxFold)]));
        
    for id = 1:length(imgFileTest)
        
        imgLoc = fullfile(imgFolderTest(id,idxFold),imgFileTest(id,idxFold));
        imgName = niftiread(char(imgLoc));
        imginfo = niftiinfo(char(imgLoc));
        
        file = char(imgFileTest(id,idxFold));        
        pid = regexp(file,'_','split');
        fname = pid{3};
        
        predLblName = ['predictedLbl_brats',fname]; 
        predDir = fullfile(destination,'predicted',['predictedLabel-fold' num2str(idxFold)],predLblName);
                
        padVol1 = padarray(imgName,[0 0 1],0);
        padVol2 = padarray(padVol1,[0 0 3],0,'pre');
        predictedLabel = semanticseg(padVol2,net,'ExecutionEnvironment','gpu','outputtype','uint8');
        unpadPred = predictedLabel(1:end-0,1:end-0,4:end-2);
        
        % save preprocessed data to folders
        niftiwrite(single(unpadPred), predDir);
                               
        id = id + 1;
        
    end
end
