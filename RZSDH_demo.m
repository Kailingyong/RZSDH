
clear;clc

addpath data;
addpath utils;

 %% 数据处理
 %% load data
load './data/mirflickr25k_1.mat'
load './data/mirflickr_A.mat'
mirflickr_attributes = y3;


XTrain = I_tr;  YTrain = T_tr;
XTest  = I_te;  YTest  = T_te;
LTrain = L_tr;  LTest  = L_te;


L_tr_Matrix = L_tr;
L_te_Matrix = L_te;


XTest  = bsxfun(@minus, XTest, mean(XTrain, 1)); 
XTrain = bsxfun(@minus, XTrain, mean(XTrain,1));
YTest  = bsxfun(@minus, YTest, mean(YTrain, 1));    
YTrain = bsxfun(@minus, YTrain, mean(YTrain,1));

[XKTrain,XKTest] = Kernelize(XTrain, XTest, 300) ; [YKTrain,YKTest]=Kernelize(YTrain, YTest, 300);
XKTest = bsxfun(@minus, XKTest, mean(XKTrain, 1)); XKTrain = bsxfun(@minus, XKTrain, mean(XKTrain, 1));
YKTest = bsxfun(@minus, YKTest, mean(YKTrain, 1)); YKTrain = bsxfun(@minus, YKTrain, mean(YKTrain, 1));


run = 10;
for i = 1 : run
    tic
fprintf('Unseen classes:\n');
load seenClass.mat;
load unseenClass.mat;
 unseenClass
 
k=1;
 k1=1;
for j=1:size(L_tr,1)
    if L_tr(j,unseenClass(:,1))==1 || L_tr(j,unseenClass(:,2))==1|| L_tr(j,unseenClass(:,3))==1|| L_tr(j,unseenClass(:,4))==1|| L_tr(j,unseenClass(:,5))==1
       index_unseen_in_tr(k,:) = j;
       k=k+1;
    else
       index_seen_in_tr(k1,:) = j;
       k1=k1+1; 
    end 
end

 k=1;
 k1=1;
for j=1:size(L_te,1)
    if L_te(j,unseenClass(:,1))==1 || L_te(j,unseenClass(:,2))==1|| L_te(j,unseenClass(:,3))==1|| L_te(j,unseenClass(:,4))==1|| L_te(j,unseenClass(:,5))==1
       index_unseen_in_te(k,:) = j;
       k=k+1;
    else
       index_seen_in_te(k1,:) = j;
       k1=k1+1; 
    end 
end


%% centralization

% train data of seen class. same as retrieal data
X1_SR = XKTrain(index_seen_in_tr,:);
X2_SR = YKTrain(index_seen_in_tr,:);
L_SR = L_tr_Matrix(index_seen_in_tr,:);

X1_SQ = XKTest(index_seen_in_te,:);
X2_SQ = YKTest(index_seen_in_te,:);
L_SQ = L_te_Matrix(index_seen_in_te,:);

% data split of unseen data
X1_UR = XKTrain(index_unseen_in_tr,:);
X2_UR = YKTrain(index_unseen_in_tr,:);
L_UR = L_tr_Matrix(index_unseen_in_tr,:);

X1_UQ = XKTest(index_unseen_in_te,:);
X2_UQ = YKTest(index_unseen_in_te,:);
L_UQ = L_te_Matrix(index_unseen_in_te,:);


S = mirflickr_attributes(seenClass,:);

%% 对训练集的标签加上噪声
L_SR1 = L_SR(:,seenClass);
[LTrain1,LTrain2,LTrain3,LTrain4] = Noiselabel_function_3(L_SR1);



%% 参数
param.alphe1  = 1e-3; param.alphe2 = 1e-3;
param.beta1  = 1e-2; param.beta2  = 1e-2;
param.thea  = 1e-1; param.lambda = 1e-1;
param.mu = 1e-3; param.gamma = 1e-2;
param.sf = 0.05; param.etaX = 0.5;
param.etaY = 0.5; 

param.lambda_c = 1e-3; param.mu1 = 1e-2;
param.rho = 0.01;



param.iter = 5;
nbitset  = 16;

eva_info = cell(1,length(nbitset));
for bit = 1:length(nbitset) 
    %% 无噪声
   param.nbits = nbitset(bit);
   [B, P1, P2, t1, t2] = RZSDH(X1_SR, X2_SR, LTrain1, param, S');
    
    %% RZSDH
    rBX = [sign(P1 * X1_UR' - t1 * ones(1,size(X1_UR,1)))';B'];
    qBX = sign(P1 * X1_UQ' - t1 * ones(1,size(X1_UQ,1)))';
    rBY = [sign(P2 * X2_UR' - t2 * ones(1,size(X2_UR,1)))';B'];
    qBY = sign(P2 * X2_UQ' - t2 * ones(1,size(X2_UQ,1)))';

    rBX = (rBX > 0);
    qBX = (qBX > 0);
    rBY = (rBY > 0);
    qBY = (qBY > 0);
    
    fprintf('\nText-to-Image Result:\n');
    mapTI = map_rank1(qBY, rBX,[L_UR; L_SR], L_UQ);
    map1(i,1) = mapTI(end);

    
    fprintf('Image-to-Text Result:\n');
    mapIT = map_rank1(qBX, rBY,[L_UR; L_SR], L_UQ);
    map1(i,2) =  mapIT(end);
   

 %% 0.2的噪声
   param.nbits = nbitset(bit);
   [B, P1, P2, t1, t2] = RZSDH(X1_SR, X2_SR, LTrain2, param, S');
    
    
    %% RZSDH
    rBX = [sign(P1 * X1_UR' - t1 * ones(1,size(X1_UR,1)))';B'];
    qBX = sign(P1 * X1_UQ' - t1 * ones(1,size(X1_UQ,1)))';
    rBY = [sign(P2 * X2_UR' - t2 * ones(1,size(X2_UR,1)))';B'];
    qBY = sign(P2 * X2_UQ' - t2 * ones(1,size(X2_UQ,1)))';

    rBX = (rBX > 0);
    qBX = (qBX > 0);
    rBY = (rBY > 0);
    qBY = (qBY > 0);
    
    fprintf('\nText-to-Image Result:\n');
    mapTI = map_rank1(qBY, rBX,[L_UR; L_SR], L_UQ);
    map2(i,1) = mapTI(end);

    
    fprintf('Image-to-Text Result:\n');
    mapIT = map_rank1(qBX, rBY,[L_UR; L_SR], L_UQ);
    map2(i,2) =  mapIT(end);

   

 %% 0.4的噪声
   param.nbits = nbitset(bit);
   [B, P1, P2, t1, t2] = RZSDH(X1_SR, X2_SR, LTrain3, param, S');

    
    %% RZSDH
    rBX = [sign(P1 * X1_UR' - t1 * ones(1,size(X1_UR,1)))';B'];
    qBX = sign(P1 * X1_UQ' - t1 * ones(1,size(X1_UQ,1)))';
    rBY = [sign(P2 * X2_UR' - t2 * ones(1,size(X2_UR,1)))';B'];
    qBY = sign(P2 * X2_UQ' - t2 * ones(1,size(X2_UQ,1)))';

    rBX = (rBX > 0);
    qBX = (qBX > 0);
    rBY = (rBY > 0);
    qBY = (qBY > 0);
    
    fprintf('\nText-to-Image Result:\n');
    mapTI = map_rank1(qBY, rBX,[L_UR; L_SR], L_UQ);
    map3(i,1) = mapTI(end);

    
    fprintf('Image-to-Text Result:\n');
    mapIT = map_rank1(qBX, rBY,[L_UR; L_SR], L_UQ);
    map3(i,2) =  mapIT(end);

 %% 0.6的噪声
   param.nbits = nbitset(bit);
   [B, P1, P2, t1, t2] = RZSDH(X1_SR, X2_SR, LTrain4, param, S');
    
    %% RZSDH
    rBX = [sign(P1 * X1_UR' - t1 * ones(1,size(X1_UR,1)))';B'];
    qBX = sign(P1 * X1_UQ' - t1 * ones(1,size(X1_UQ,1)))';
    rBY = [sign(P2 * X2_UR' - t2 * ones(1,size(X2_UR,1)))';B'];
    qBY = sign(P2 * X2_UQ' - t2 * ones(1,size(X2_UQ,1)))';

    rBX = (rBX > 0);
    qBX = (qBX > 0);
    rBY = (rBY > 0);
    qBY = (qBY > 0);
    
    fprintf('\nText-to-Image Result:\n');
    mapTI = map_rank1(qBY, rBX,[L_UR; L_SR], L_UQ);
    map4(i,1) = mapTI(end);

    
    fprintf('Image-to-Text Result:\n');
    mapIT = map_rank1(qBX, rBY,[L_UR; L_SR], L_UQ);
    map4(i,2) =  mapIT(end);

end
end
map11 = mean(map1)
map22 = mean(map2)
map33 = mean(map3)
map44 = mean(map4)
