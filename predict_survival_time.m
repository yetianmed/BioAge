% This is an example script to show the analysis of survial time prediction where
% age at assessement, sex and organ age gaps are included in the model.
% Other predictors such as existing diagnoses and lifestyle factors can be added.

clear
load data.mat T age_at_assessment sex age_gap censored

%%%% censored: mortality status. 1=non-deceased; 0=deceased.
%           dimension: nSubjects x 1

%%%% T: post-assessment survival time
%       for deceased people, T=number of days between the date of death and
%       the date of organ function assessment.
%       for non-deceased people, T=number of days between the date of
%       mortality ascertainment and the date of organ function assessment.
%       dimension: nSubjects x 1

% Check some numbers
fprintf('Number of subjects: n=%d\n',size(x,1))
ind_dead=find(~censored);
fprintf('Death: n=%d\n',length(ind_dead));
fprintf('Mortality rate %.3f\n',length(ind_dead)/size(x,1));
ind_alive=find(censored);
T=age_final;
hf=figure; hf.Color='w';
subplot(1,2,1);
hist(T(ind_dead),100); ylabel('Deceased'); xlabel('Years');
subplot(1,2,2)
hist(T(ind_alive),100); ylabel('Living'); xlabel('Years');

% Prepare the data for model training
age_at_assessment=zscore(age_at_assessment);
age_gap=zscore(age_gap);
x=double([age_at_assessment,sex,age_gap]);

% survival years post assessment
T=T/365; %convert from days to years

% Define 5-year and 10-year survival
t=[5,10]; 
Folds=10; % k-fold cross-validation

% Bootstraps for 95% CI estimation 
N=size(x,1);
nboot=100;
ind_boot=zeros(N,nboot);
for nn=1:nboot
    ind_boot(:,nn)=randsample(N,N,true);
end

% add non-bootstrapped sample to the last column
ind_boot=[ind_boot,[1:N]'];

% area under the curve (accuracy)
auc=zeros(size(ind_boot,2),length(t)); 
for nn=1:size(ind_boot,2)
    
        x_boot=x(ind_boot(:,nn),:);
        T_boot=T(ind_boot(:,nn));
        censored_boot=censored(ind_boot(:,nn));
        
    for j=1:length(t)
        
        % need to check those T<t only includes deceased people   
        % outcome
        c{j}=T_boot<t(j); % In our sample, all non-deceased people had been lived more than 10 years

        % 10-fold cross-validation
        cv=cvpartition(length(c{j}),'KFold',Folds);
        ypred{j}=zeros(length(c{j}),1);
        for i=1:Folds
            ind_train=find(cv.training(i));
            ind_test=find(cv.test(i));
            
            %Logistic regression
            [b{i},dev,stats{i}]=mnrfit(x_boot(ind_train,:),categorical(c{j}(ind_train)));
            ypred{j}(ind_test)=[ones(length(ind_test),1),x_boot(ind_test,:)]*b{i};
            fprintf('Fold %d of %d\n',i,Folds);
        end
        B{j}=b;
        
        [r,ind_srt]=sort(ypred{j});
        TP=sum(c{j}); TN=sum(~c{j});
        false_pos_rate=cumsum(~c{j}(ind_srt))/TN;
        true_pos_rate=cumsum(c{j}(ind_srt))/TP;
        auc(nn,j)=trapz(false_pos_rate,true_pos_rate);
        fprintf('Bootstrapped sample %d,auc=%0.2f%%\n',nn,auc(nn,j)*100);
    end
end
