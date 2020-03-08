% SeqSimRand_v1.m
% 
% G. Elliott Wimmer, Yunzhe Liu 2019
% 
% simulate data and test for sequenceness relationship to behavior given
% noise input
% 
% requires fitglme function
% 
% SeqSimRand.m
% 
% v0:  initial version
% 
% v1:  clean up options, add across-subject correlation


%% initial set up
clear

% set number of simulations
nsim =        1000;
simstart =       1;
fprintf(['\n' 'Running ' num2str(nsim) ' simulations.' '\n\n'])


% optional:  set directory
analysisdir='/data1/memstructmeg/analysism';
addpath(analysisdir);
cd(analysisdir);

% set variables
ncat =           6;
nstate =         4; % for transition matrix calculation
nepi =           8; % 8 episodes
ntrials =      220; % even number of trials
nTr =      ntrials;
maxLag =        60; % for sim, evaluate cross-correlation up to 600 ms
maxLagwin =     35; % actual window of 350 ms is checked
nSensors =     273; % 275-2 missing
nTrainPerStim = 18; % how many training examples for each stimulus
nNullExamples = nTrainPerStim*ncat; % how many null examples to use
nSamples =     353; % (3.67 - 0.160 s) unlabelled data to predict

%%% set l1 param
l1param =        2; % fix at 2
fprintf(['\nL1param = ' num2str(l1param) ' \n'])

%%% set number of subjects and regular-perf group
nSubj =         25; % 25
nSubjincl =     18; % use 18 regular-performing subjects in regressions 
fprintf(['\n' 'Running simulated n=' num2str(nSubj) '.' '\n'])


% set subject variable
subjall = [];
for iSj = 1:nSubj
    subjtrials = ones(ntrials,1)*iSj;
    subjall = [subjall; subjtrials];
end
clear subjtrials

% set randomly excluded trials ~= artifact MEG trial data
memtrial = ntrials-nSubj*2+2:2:ntrials;
memtrialincl = NaN(nSubj,ntrials);
for iSj = 1:nSubj
    subjtrials = [ones(memtrial(iSj),1); zeros(ntrials-memtrial(iSj),1)];
    subjtrials(:,2) = rand(ntrials,1);
    subjtrials = sortrows(subjtrials,2);
    subjtrials = subjtrials(:,1);
    memtrialincl(iSj,:) = subjtrials;
end
clear memtrial
clear subjtrials

% set fixed after/before condition
afterbefore = repmat([ones(10,1); -ones(10,1)],(ntrials/20)*nSubj,1);

% set episode state category order list (from actual episodes)
catlist = [5,2,1,4;
    3,4,5,6;
    2,4,6,1;
    4,2,1,3;
    1,2,5,3;
    6,3,1,5;
    4,1,6,2;
    2,3,6,5];

% set episode cued on each trial (here sampling each in order)
epi2state = repmat((1:nepi)',ceil(ntrials/nepi),1);
epi2state = epi2state(1:ntrials);

% set storage
pvala = NaN(nsim,2);
pvalb = NaN(nsim,2);
tstata = NaN(nsim,2);
tstatb = NaN(nsim,2);
corrinddifflosoa = NaN(nsim,2);
corrinddifflosob = NaN(nsim,2);
corrinddiffmaxa = NaN(nsim,2);
corrinddiffmaxb = NaN(nsim,2);
simpeaklosotime = NaN(nsim,nSubj);
simpeakmaxtime = NaN(nsim,1);
dtpdiffaftersim = NaN(nsim,nSubj,maxLagwin+1);
dtpdiffbeforesim = NaN(nsim,nSubj,maxLagwin+1);


input('\nConfirm the above\n')
fprintf('Ok...\n\n');


tic


for iP = simstart:nsim
    
    % set storage
    sf = cell(nSubj,1);
    sb = cell(nSubj,1);
    
    % set correct vector across subjects; preserve regular order across sims
    memperf = round([0.6:.02:1 0.76 0.78 0.82 0.84]'.*ntrials);
    memperf = sortrows(memperf);
    memthresh = memperf(nSubjincl);
    memperf = memperf(:,1);
    correct = [];
    grouplow = [];
    for iSj = 1:nSubj
        subjtrials = [ones(memperf(iSj),1); zeros(ntrials-memperf(iSj),1)];
        subjtrials(:,2) = rand(ntrials,1);
        subjtrials = sortrows(subjtrials,2);
        subjtrials = subjtrials(:,1);
        correct = [correct; subjtrials];
        if memperf(iSj)<=memthresh+1
            subjgroup = ones(ntrials,1);
        else
            subjgroup = -ones(ntrials,1);
        end
        grouplow = [grouplow; subjgroup];
    end
    clear subjtrials
    clear subjgroup
    
    
    % loop across subjects
    for iSj=1:nSubj
        
        
        disp(['iSj = ' num2str(iSj) ';  permutation = ' num2str(iP)])
        
        %% simulate category training data and train classifiers
        
        % generate dependence of the sensors
        A = randn(nSensors);
        [U,~] = eig((A+A')/2);
        covMat = U*diag(abs(randn(nSensors,1)))*U';
        
        % generate the true patterns
        commonPattern = randn(1,nSensors);
        patterns = repmat(commonPattern, [ncat 1]) + randn(ncat, nSensors);
        
        % make training data
        trainingData = 4*randn(nNullExamples+ncat*nTrainPerStim, nSensors) + [zeros(nNullExamples,nSensors); ...
            repmat(patterns, [nTrainPerStim 1])];
        trainingLabels = [zeros(nNullExamples,1); repmat((1:ncat)', [nTrainPerStim 1])];
        
        % train classifiers on training data
        betasens = NaN(nSensors, ncat);
        intercepts = NaN(1,ncat);
        
        for iC=1:ncat
            [betasens(:,iC), fitInfo] = lassoglm(trainingData, trainingLabels==iC, 'binomial', 'Alpha', 1, 'Lambda', 0.002, 'Standardize', false);
            intercepts(iC) = fitInfo.Intercept;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % set storage
        sf{iSj} = NaN(nTr, maxLag+1);
        sb{iSj} = NaN(nTr, maxLag+1);
        
        
        for iTr=1:nTr
            
            %% core trial: loop by lasso
            for vec=l1param
                
                
                %% create X data and make predictions with trained models
                X = NaN(nSamples, nSensors);
                X(1,:) = randn([1 nSensors]);
                
                % add autocorrelation to data
                % actual data autocorrelation (r = 0.499); acorr = 0.625 yields t:t-1 autocorrelation of r~=0.52 in applied classifiers
                acorr = 0.65;
                for iT=2:nSamples
                    X(iT,:) = acorr*(X(iT-1,:) + mvnrnd(zeros(1,nSensors), covMat)); % add dependence of the sensors
                end
                
                % make predictions with trained models
                preds = 1./(1+exp(-(X*betasens + repmat(intercepts, [nSamples 1]))));
                
                % get current episode identity
                episodenow=epi2state(iTr);
                
                if memtrialincl(iSj,iTr)==0
                    % skip trials with meg "artifacts"
                    continue
                end
                
                X = preds;
                
                % sort to extract correct order of 4 categories
                catsortnow = catlist(episodenow,:);
                
                % shuffle to match tx matrix
                X = X(:,catsortnow);
                
                %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % regression
                nbins=maxLag+1;
                warning off
                dm=[toeplitz(X(:,1),[zeros(nbins,1)])];
                dm=dm(:,2:end);
                
                for kk=2:nstate
                    temp=toeplitz(X(:,kk),[zeros(nbins,1)]);
                    temp=temp(:,2:end);
                    dm=[dm temp];
                end
                
                warning on
                
                Y=X;
                
                betas = NaN(nstate*maxLag, nstate);
                
                % minimizes fluctuations
                for ilag=1:maxLag
                    zinds = (1:maxLag:nstate*maxLag) + ilag - 1;
                    
                    for xstate=1:nstate
                        temp = pinv([dm(:,zinds(xstate)) ones(length(dm(:,zinds)),1)])*Y;
                        betas(zinds(xstate),:)=temp(1:end-1,:);
                    end
                end
                
                
                %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                betasnbins64=reshape(betas,[maxLag nstate^2]);
                
                % true transition matrix across all trials given that
                % categories are sorted per-trial above
                TF = [0,1,0,0
                      0,0,1,0
                      0,0,0,1
                      0,0,0,0];
                
                T1 = TF;
                T2 = T1'; % backwards is transpose of forwards
                
                % combined fwd and bkw sequenceness in the same regression model
                bbb=pinv([T1(:) T2(:) squash(eye(nstate))])*(betasnbins64');
                % do z-score
                sf{iSj}(iTr,2:end)=zscore(bbb(1,:));
                sb{iSj}(iTr,2:end)=zscore(bbb(2,:));
                
            end
        end
        
    end
    
    
    save tempsimall
    
    %% extract sequenceness
    if iscell(sf) && iscell(sb)
        sfcell = sf;
        sbcell = sb;
        sf = cell2mat(sf);
        sb = cell2mat(sb);
    end
    
    % extract and  mean-correct for multilevel regressions
    sftemp = [];
    sfmeancorr = [];
    sbtemp = [];
    sbmeancorr = [];
    for iSj = 1:nSubj
        % forward
        temp = sfcell{iSj};
        sftemp = [sftemp; temp];
        temp = temp-nanmean(temp);
        sfmeancorr = [sfmeancorr; temp];
        % backward
        temp = sbcell{iSj};
        sbtemp = [sbtemp; temp];
        temp = temp-nanmean(temp);
        sbmeancorr = [sbmeancorr; temp];
    end
    clear temp
    
    % get subject list
    subjlist = unique(subjall);
    
    % get fwd-bkw sequenceness measure per trial
    sdiffall = sftemp-sbtemp;
    sdiffall = sdiffall(:,1:maxLagwin+1);
    
    % get mean seq per subject for calculating peak after condition correct trial replay
    dtpdiffafter = NaN(nSubj,maxLagwin+1);
    dtpdiffbefore = NaN(nSubj,maxLagwin+1);
    for iSj = 1:nSubj
        dtpdiffafter(iSj,:) = nanmean(sdiffall(correct==1 & afterbefore==1 & subjall==subjlist(iSj),:));
        dtpdiffbefore(iSj,:) = nanmean(sdiffall(correct==1 & afterbefore==-1 & subjall==subjlist(iSj),:));
    end
    
    %% get leave-one-out cross-validation peak timepoint per subject
    maxtime = 35;
    tpstart = 5; % 5 = 40 ms lag
    dtpxmean = NaN(nSubj,maxtime);
    dtppeaklosotime = NaN(nSubj,1);
    dtppeak = [];
    for iSj = 1:nSubj
        
        dtptemp = dtpdiffafter(:,1:maxtime);
        dtptemp(iSj,:) = NaN;
        dtpxmean(iSj,:) = abs(nanmean(dtptemp));
        
        % timelag -1 to get actual time lag, but account for this later
        subjpeak = (find(dtpxmean(iSj,:)==max(dtpxmean(iSj,tpstart:maxtime)))-1);
        dtppeaklosotime(iSj,1) = subjpeak(1)*10;
        
        % get peak value +-10 ms to data table (subjpeak is -10 ms before actual peak)
        if subjpeak==4 % subjpeak of 4 = 30ms; only start at 40ms
            subjlagrange = subjpeak+1:subjpeak+2;
        else
            subjlagrange = subjpeak:subjpeak+2;
        end
        peakfwd = nanmean(sfmeancorr(subjall==subjlist(iSj),subjlagrange),2);
        peakbkw = nanmean(sbmeancorr(subjall==subjlist(iSj),subjlagrange),2);
        
        % concat each subject's peakfwd and peakbkw sequenceness onto group
        dtppeak = [dtppeak; [peakfwd peakbkw]];
    end
    
    %% get max abs lag for individual difference correlations
    dtptemp = dtpdiffafter(:,1:maxtime);
    dtptemp(:,1:tpstart-1) = NaN;
    dtpxmean = abs(nanmean(dtptemp));
    %%% timelag -1 to get actual time lag, but account for this later
    [~, maxpeak] = max(dtpxmean);
    maxpeak = maxpeak - 1;
    dtppeakmaxtime(:,1) = maxpeak*10;
    
    
    %% create data table for multilevel regression (fitglme)
    data = table(subjall, grouplow, correct, afterbefore, dtppeak(:,1), dtppeak(:,2));
    % add labels to table
    data.Properties.VariableNames{1} = 'subj';
    data.Properties.VariableNames{2} = 'grouplow';
    data.Properties.VariableNames{3} = 'acc';
    data.Properties.VariableNames{4} = 'afterbefore';
    data.Properties.VariableNames{5} = 'seqfwd';
    data.Properties.VariableNames{6} = 'seqbkw';
    
    % subset data
    dafter = data(data.afterbefore==1,:);
    dbefore = data(data.afterbefore==-1,:);
    % get after condition in regular performance after group
    aNafter = dafter(dafter.grouplow>0,:);
    % get before condition in regular performance before group
    bNbefore = dbefore(dbefore.grouplow>0,:);
    
    %% get data for inddiff regressions
    memperf = memperf/max(memperf);
    dtpcorrinddifflosoa = NaN(nSubj,1);
    dtpcorrinddifflosob = NaN(nSubj,1);
    dtpcorrinddiffmaxa = NaN(nSubj,1);
    dtpcorrinddiffmaxb = NaN(nSubj,1);
    for iSj = 1:nSubj
        if subjpeak==4 % subjpeak of 4 = 30ms; only start at 40ms
            subjlagrange = subjpeak+1:subjpeak+2;
        else
            subjlagrange = subjpeak:subjpeak+2;
        end
        dtpcorrinddifflosoa(iSj,1) = nanmean(dtpdiffafter(iSj,subjlagrange),2);
        dtpcorrinddifflosob(iSj,1) = nanmean(dtpdiffbefore(iSj,subjlagrange),2);
        if (dtppeakmaxtime/10)==4 % grouppeak of 4 = 30ms; only start at 40ms
            grouplagrange = dtppeakmaxtime/10+1:dtppeakmaxtime/10+2;
        else
            grouplagrange = dtppeakmaxtime/10:dtppeakmaxtime/10+2;
        end
        dtpcorrinddiffmaxa(iSj,1) = nanmean(dtpdiffafter(iSj,grouplagrange),2);
        dtpcorrinddiffmaxb(iSj,1) = nanmean(dtpdiffbefore(iSj,grouplagrange),2);
    end
    
    %% run inddiff regressions; for sim assume after performance = before performance
    % max lag preferred
    [rafter, pafter] = corrcoef(memperf,dtpcorrinddiffmaxa);
    [rbefore, pbefore] = corrcoef(memperf,dtpcorrinddiffmaxb);
    corrinddiffmaxa(iP,1:2) = [rafter(2) pafter(2)];
    corrinddiffmaxb(iP,1:2) = [rbefore(2) pbefore(2)];
    % (leave-one-subject-out lag non-preferred)
    [rafter, pafter] = corrcoef(memperf,dtpcorrinddifflosoa);
    [rbefore, pbefore] = corrcoef(memperf,dtpcorrinddifflosob);
    corrinddifflosoa(iP,1:2) = [rafter(2) pafter(2)];
    corrinddifflosob(iP,1:2) = [rbefore(2) pbefore(2)];
    
    
    %% run multilevel regressions (roughly equivalent to primary models run in R; note that R is more reliable for precise stats)
    rowstat =    3;
    colpval =    6;
    coltstat =   4;
    disp('fitting after model')
    lmea = fitglme(aNafter,'acc ~ seqfwd + seqbkw + (1 | subj) + (seqfwd | subj) + (seqbkw | subj)');
    disp('fitting before model')
    lmeb = fitglme(bNbefore,'acc ~ seqfwd + seqbkw + (1 | subj) + (seqfwd | subj) + (seqbkw | subj)');
    % extract coefficients
    lmeacoef = dataset2cell(lmea.Coefficients);
    lmebcoef = dataset2cell(lmeb.Coefficients);
    % store p-values from after and before models
    pvala(iP,1:2) = [lmeacoef{rowstat,colpval} lmeacoef{rowstat+1,colpval}];
    pvalb(iP,1:2) = [lmebcoef{rowstat,colpval} lmebcoef{rowstat+1,colpval}];
    
    tstata(iP,1:2) = [lmeacoef{rowstat,coltstat} lmeacoef{rowstat+1,coltstat}];
    tstatb(iP,1:2) = [lmebcoef{rowstat,coltstat} lmebcoef{rowstat+1,coltstat}];
    
    % optional:  store selected per-subject peak lag and mean fwd-bkw difference
    simpeaklosotime(iP,:) = dtppeaklosotime';
    simpeakmaxtime(iP,1) = dtppeakmaxtime(1);
    dtpdiffaftersim(iP,:,:) = dtpdiffafter;
    dtpdiffbeforesim(iP,:,:) = dtpdiffbefore;
    
    %% optional:  save per iteration
%     if rem(iP,500)
        save(['simrandall_ac65_p1p' num2str(iP) '_rand' num2str(round(rand,2)*100)],'pval*','tstat*','simpeaklosotime','simpeakmaxtime','dtpdiffaftersim','dtpdiffbeforesim','corrinddiff*','memperf');
%     end
    toc
end


%% individual differences analyses:  compute false positive rate and adjusted 5% threshold

% empirical permutation-derived adjusted 5% threshold
threshadjinddiffafter = prctile(corrinddiffmaxa(:,2),5);
threshadjinddiffbefore = prctile(corrinddiffmaxb(:,2),5);

% empirical permutation-derived false positive rate
fposrateinddiffafter = nanmean(corrinddiffmaxa(:,2)<0.05);
fposrateinddiffbefore = nanmean(corrinddiffmaxb(:,2)<0.05);

% likelihood of finding a difference between after and before correlations >= 0.8
pvalinddiffconj = abs(corrinddiffmaxa(:,1)-corrinddiffmaxb(:,1))>=0.8; % actual data correlation diff ~ 0.8 (data: r>0.40, r>-0.40)
pvalinddiffconj = sum(pvalinddiffconj)./length(pvalinddiffconj);



%% trial-by-trial analyses:  compute false positive rate and adjusted 5% threshold

% empirical permutation-derived adjusted 5% threshold
threshadjfwdafter = prctile(pvala(:,1),5);
threshadjbkwafter = prctile(pvala(:,2),5);
threshadjfwdbkwafter = (prctile(pvala(:),5));
threshadjfwdbefore = prctile(pvalb(:,1),5);
threshadjbkwbefore = prctile(pvalb(:,2),5);
threshadjfwdbkwbefore = (prctile(pvalb(:),5));

% empirical permutation-derived false positive rate
fposratefwdafter = nanmean(pvala(:,1)<0.05);
fposratebkwafter = nanmean(pvala(:,2)<0.05);
fposratefwdbkwafter = nanmean(pvala(:)<0.05);
fposratefwdbefore = nanmean(pvalb(:,1)<0.05);
fposratebkwbefore = nanmean(pvalb(:,2)<0.05);
fposratefwdbkwbefore = nanmean(pvalb(:)<0.05);


%% save
save(['simrandall_full_ac65_rand' num2str(round(rand,2)*100)],'pval*','tstat*','simpeaklosotime','simpeakmaxtime','dtpdiffaftersim','dtpdiffbeforesim','corrinddiff*','thresh*','fpos*','memperf');



%% plot individual differences false positive rate (% of results per 0.05 bin)
dataafter = corrinddiffmaxa(:,2);
databefore = corrinddiffmaxb(:,2);
dataafter = hist(dataafter,20)./length(dataafter);
databefore = hist(databefore,20)./length(databefore);

figure,
x0=50;
y0=50;
width=800;
height=540;
set(gcf,'units','points','position',[x0,y0,width,height]);
pVal = 0.05:0.05:1;
colorblue = [0 0 255]/255;
colorcyan = [0 255 200]/255;
shadedErrorBar(pVal, dataafter, dataafter/100000,{'color', colorcyan, 'LineWidth', 4},0); hold on
shadedErrorBar(pVal, databefore, databefore/100000,{'color', colorblue, 'LineWidth', 4},0); hold on
title(['a+b (fwd-bkw) ' num2str(length(corrinddiffmaxa)) ' inddiff randperm; <5% after ' num2str(round(mean(corrinddiffmaxa(:,2)<0.05),4)) '; before ' num2str(round(mean(corrinddiffmaxb(:,2)<0.05),4))]);
xlabel('p-value bin')
ylabel('probability')
set(gca,'FontSize',18)
hline = refline([0 0.05]);
hline.Color = 'black';
xlim([0.05 1])
set(gca,'Xtick',0.05:0.05:1)
xtickangle(gca,-45)
ylim([0 .1])
a = gca;
set(a,'box','off','color','none') % set box property to off and remove background color
b = axes('Position',get(a,'Position'),'box','on','xtick',[],'ytick',[]); % create new, empty axes with box but without ticks
axes(a) % set original axes as active

figname = (['mem_inddiff_randpermmax_permn' num2str(size(pvala,1))]);
saveas(gcf,figname,'fig')
saveas(gcf,figname,'png')



%% plot trial-by-trial multilevel model false positive rate (% of results per 0.05 bin)
dataafter = hist(pvala(:),20)./length(pvala(:));
databefore = hist(pvalb(:),20)./length(pvalb(:));

figure,
x0=50;
y0=50;
width=800;
height=540;
set(gcf,'units','points','position',[x0,y0,width,height]);
pVal = 0.05:0.05:1;
colorblue = [0 0 255]/255;
colorcyan = [0 255 200]/255;
shadedErrorBar(pVal, dataafter, dataafter/100000,{'color', colorcyan, 'LineWidth', 4},0); hold on
shadedErrorBar(pVal, databefore, databefore/100000,{'color', colorblue, 'LineWidth', 4},0); hold on
title(['a+b fwd+bkw ' num2str(length(pvala(:,1))) ' randperm; <5% after ' num2str(round(sum(pvala(:)<0.05)/length(pvala(:)),4)) '; before ' num2str(round(sum(pvalb(:)<0.05)/length(pvalb(:)),4))]);
xlabel('p-value bin')
ylabel('probability')
set(gca,'FontSize',18)
hline = refline([0 0.05]);
hline.Color = 'black';
xlim([0.05 1])
set(gca,'Xtick',0.05:0.05:1)
xtickangle(gca,-45)
ylim([0 .1])
a = gca;
set(a,'box','off','color','none') % set box property to off and remove background color
b = axes('Position',get(a,'Position'),'box','on','xtick',[],'ytick',[]); % create new, empty axes with box but without ticks
axes(a) % set original axes as active

figname = (['mem_randpermloso_permn' num2str(size(pvala,1))]);
saveas(gcf,figname,'fig')
saveas(gcf,figname,'png')