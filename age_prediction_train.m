clear
close all

% phenotype data for model training
load data_train.mat X age sex

% X: phenotype matrix (nSubjects x nPhenotypes)
% age: chronological age of healthy individuals included in model training (nSubjects x 1 )
% sex: female=0; male=1 (nSubjects x 1 )

% Train the model for male and female seperately
Gender=1; %0=female;1=male
X=X(sex==Gender,:);
age=age(sex==Gender);

% Create train and test samples using 20 folds cvpartition
K=20; % K-fold 
C=cvpartition(size(X,1),'KFold',K);
age_predic=zeros(size(age));
r=zeros(1,K);
for k=1:K
    
    % Train
    yTrain=age(C.training(k));
    xTrain=X(C.training(k),:);
    
    % Test
    yTest=age(C.test(k));
    xTest=X(C.test(k),:);
    ind_test=C.test(k);
    
    fprintf('Training fold %d...\n',k)
    Mdl = fitrsvm(xTrain,yTrain,...
        'Standardize',true,...
        'KernelFunction','linear',...
        'KernelScale','auto',...
        'Verbose',0,'CacheSize',...
        'maximal');
    
    % Predict
    yhat=predict(Mdl,xTest);
    age_predic(ind_test)=yhat;
end

% model performance
age_real=age;
r_predic=corr(age_real,age_predic);
mae=mean(abs(age_predic-age_real));% mean absolute error
fprintf('Prediction outcome: correlation r=%.2f mae=%.2f\n',r_predic,mae)

%%% age bias correction %%%
gap_resid=zeros(size(age_real));
for k=1:K
    
    % Train set
    ind_train=find(C.training(k));
    ytrain=age_real(ind_train);% chronological age 
    ytrain_predic=age_predic(ind_train);% predicted age
    gap=ytrain_predic-ytrain;% age gap 
    
    % age dependence of gap 
    b=glmfit(ytrain,gap);
    
    % Test set
    ind_test=find(C.test(k));
    ytest=age_real(ind_test);% chronological age 
    ytest_predic=age_predic(ind_test);% predicted age
    gap=ytest_predic-ytest;% age gap
    yfit=glmval(b,ytest,'identity'); % use beta estimated from the train set
    gap_resid(ind_test)=gap-yfit;% corrected age gap
end

% beta coefficient 
% this is to be used for age-bias correction in individuals with disease (see test_svm.m)
gap=age_predic-age_real;
beta=glmfit(age_real,gap);

% Train the model with all healthy subjects
yTrain=age;
xTrain=X;
fprintf('Gender=%d,Training on the whole sample n=%d\n',Gender,length(age));
Mdl=fitrsvm(xTrain,yTrain,...
    'Standardize',true,...
    'KernelFunction','linear',...
    'KernelScale','auto',...
    'Verbose',0,'CacheSize',...
    'maximal');
save(['model_sex',num2str(Gender),'.mat'],'Mdl','age_real',...
    'age_predic','r_predic','sex','beta');






