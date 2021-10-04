clear
close all

% phenotype data in test
load data_test.mat X age sex 

% X: phenotypes matrix (nSubjects x number of phenotypes)
% age: chronological age of individuals with disease (nSubjects x 1 )
% sex: female=0; male=1 (nSubjects x 1 )

% Test the model for male and female seperately
Gender=1; %0=female;1=male
xTest=X(sex==Gender,:);
yTest=age(sex==Gender);

load (['model_sex',num2str(Gender),'.mat'],'Mdl','beta')
yhat=predict(Mdl,xTest);
r_predic=corr(yhat,yTest);
mae=mean(abs(yhat-yTest));
fprintf('Prediction outcome: correlation r=%.2f mae=%.2f\n',r_predic,mae)

% compute the age gap
age_real=yTest;
age_predic=yhat;
gap=age_predic-age_real;

% regress out age dependence using beta computed from the training set
yfit=glmval(beta,age_real,'identity');
gap_resid=gap-yfit; % corrected age gap
save(['test_sex',num2str(Gender),'.mat'],'age_real','age_predic',...
    'r_predic','mae','gap_resid');



