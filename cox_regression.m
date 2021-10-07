clear
fprintf('Cox proportional hazards regression...\n')
load survival_data.mat x T censored

%%% x: risk factors including chronological age, sex, organ age gaps,lifestyle factors, presence of
%      diagnoses etc. 
%      dimension: nSubjects x nFactors
% censored: mortality status. 1=non-deceased; 0=deceased. 
%           dimension: nSubjects x 1
% T: survival time 
%    for deceased people, T=number of days between the date of death and
%    the date of organ function assessment.
%    for non-deceased people, T=number of days between the date of
%    mortality ascertainment and the date of organ function assessment.
  
T=T/365; %convert from days to years to be consistent with the unit of age gap

fprintf('Proportion of right-censored samples: %0.2f%%\n',sum(censored)/length(censored)*100);

% chronological age and age gap can be standardised before adding to the model
[b,logl,H,stats]=coxphfit(x,T,'censoring',censored);
y=x*b;
hazard_ratio=exp(b);% hazard ratio for each factor
z=stats.z; % effect size for each factor

% compute auc
[y_srt,ind_srt]=sort(y,'descend');
TP=sum(~censored); TN=sum(censored);
false_pos_rate=cumsum(censored(ind_srt))/TN;
true_pos_rate=cumsum(~censored(ind_srt))/TP;
auc=trapz(false_pos_rate,true_pos_rate);

%Bonferroni correction
p=stats.p;
ind_sig=(p<0.05/size(x,2)); % significant factors

%%%% run bootstraps to estimate 95% CI
fprintf('Bootstrapping\n')
N=size(x,1);
nboot=100; % number of bootstraps
frst=0;
b_rnd=zeros(size(x,2),nboot);
for i=1:nboot
    ind_rnd=randsample(N,N,true);
    x_rnd=x(ind_rnd,:);
    T_rnd=T(ind_rnd);
    censored_rnd=censored(ind_rnd);
    [b_rnd(:,i),log_rnd(i),~,stats_rnd{i}]=coxphfit(x_rnd,T_rnd,'censoring',censored_rnd);
    show_progress(i,nboot,frst);frst=1;
end

% sort beta
b_rnd_srt=sort(b_rnd,2);
ci=exp([b_rnd_srt(:,5),b_rnd_srt(:,95)]);% 95% CI
se=abs(ci-hazard_ratio);

Figure=0;
if Figure
    % bar plot of hazard ratios for
    hf=figure; hf.Color='w';
    hb=bar(hazard_ratio,0.6,'r');alpha(0.5)
    hb.FaceColor = 'flat';
    % make bar white if not significant
    for i=1:size(x,2)
        if ind_sig(i)==0
            hb.CData(i,:)=[1,1,1];
        end
    end
    hold on
    % add CI
    he=errorbar([1:length(se)],hazard_ratio,se(:,1),se(:,2),'.');
    he.Color='k';
    he.LineWidth=1;
    ylabel('Hazard ratio');
    ha=gca;
    ha.YGrid='on';
    
    % effect size
    hf=figure; hf.Color='w';
    hb=bar(z,0.6,'b');
    hb.FaceColor = 'flat';
    hb.FaceAlpha=0.5;
    % make bar white if not significant
    for i=1:size(x,2)
        if ind_sig(i)==0
            hb.CData(i,:)=[1,1,1]; 
        end
    end
    ylabel('z-score');
    ha=gca;
    ha.YGrid='on';
end


