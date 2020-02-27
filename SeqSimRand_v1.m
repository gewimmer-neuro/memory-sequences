% TrySeqSim.m
% 
% YL, GEW 2019
% 
% simulate data and test for sequenceness relationship to behavior given
% noise input


% SeqSimRand.m
% 
% v0:  initial version
% 
% v1:  clean up options, add across-subject correlation


%% initial set up
clear

load simrandall_ac65_tstart40ms_p1p1_rand7

% set number of simulations
nsim =           2;
simstart =       2;
fprintf(['\n' 'Running ' num2str(nsim) ' simulations.' '\n\n'])


% optional:  set directory
analysisdir='/data1/memstructmeg/analysism';
addpath(analysisdir);
cd(analysisdir);

% set variables
ncat =           6;
nstate =         4; % for transition matrix calculation
nepi =           8; % 8 episodes
ntrials    =   220; % even number of trials
nTr =      ntrials;

maxLag =        60; % for sim, evaluate cross-correlation up to 600 ms
maxLagwin =     35; % actual window of 350 ms is checked
nSensors =     273; % 275-2 missing
nTrainPerStim = 18; % how many training examples for each stimulus
nNullExamples = nTrainPerStim*ncat; % how many null examples to use
nSamples =     353; % 3.5 sec of unlabelled data to predict

%%% set l1 param
l1param =        2; % fix at 2...
fprintf(['\nL1param = ' num2str(l1param) ' \n'])

%%% set number of subjects and regular-perf group
nSubj =         25;
nSubjincl =     18; % use 18 regular-perf in regressions 
fprintf(['\n' 'Running simulated n=' num2str(nSubj) '.' '\n'])


% set subject variable
subj = [];
for iSj = 1:nSubj
    subjtrials = ones(ntrials,1)*iSj;
    subj = [subj; subjtrials];
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
    
    % set vector for later inclusion in correct/incorrect regression analysis
    subjincl = 0;
    
    
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
                bbb=pinv([T1(:) T2(:) squash(eye(nstate))])*(betasnbins64'); % no constant *default*
                % do z-score
                sf{iSj}(iTr,2:end)=zscore(bbb(1,:));
                sb{iSj}(iTr,2:end)=zscore(bbb(2,:));
                
            end
        end
        
    end
    
    
    %% extract sequenceness
    if iscell(sf) && iscell(sb)
        sfcell = sf;
        sbcell = sb;
        sf = cell2mat(sf);
        sb = cell2mat(sb);
    end
    
    % mean-correct for multilevel regressions
    sftemp = [];
    sftemp0 = [];
    for iSj = 1:nSubj
        temp = sfcell{iSj};
        sftemp = [sftemp; temp];
        % mean-correct
        temp = temp-nanmean(temp);
        sftemp0 = [sftemp0; temp];
    end
    clear temp
    % mean-correct for multilevel regressions
    sbtemp = [];
    sbtemp0 = [];
    for iSj = 1:nSubj
        temp = sbcell{iSj};
        sbtemp = [sbtemp; temp];
        % mean-correct
        temp = temp-nanmean(temp);
        sbtemp0 = [sbtemp0; temp];
    end
    clear temp
    
    % get subject list
    subjlist = unique(subj);
    
    % get fwd-bkw sequenceness measure per trial
    sdiffall = sftemp-sbtemp;
    sdiffall = sdiffall(:,1:maxLagwin+1);
    
    % get mean seq per subject for calculating peak after condition correct trial replay
    dtpdiffafter = NaN(nSubj,maxLagwin+1);
    dtpdiffbefore = NaN(nSubj,maxLagwin+1);
    for iSj = 1:nSubj
        dtpdiffafter(iSj,:) = nanmean(sdiffall(correct==1 & afterbefore==1 & subj==subjlist(iSj),:));
        dtpdiffbefore(iSj,:) = nanmean(sdiffall(correct==1 & afterbefore==-1 & subj==subjlist(iSj),:));
    end
    
    % get leave-one-out cross-validation peak timepoint per subject
    maxtime = 35;
    tpstart = 5; % 5 = 40 ms lag!
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
        peakfwd = nanmean(sftemp0(subj==subjlist(iSj),subjpeak:subjpeak+2),2);
        peakbkw = nanmean(sbtemp0(subj==subjlist(iSj),subjpeak:subjpeak+2),2);
        
        % concat each subject's peakfwd and peakbkw sequenceness onto group
        dtppeak = [dtppeak; [peakfwd peakbkw]];
    end
    
    % and get max abs lag
    dtptemp = dtpdiffafter(:,1:maxtime);
    dtptemp(:,1:tpstart-1) = NaN;
    dtpxmean = abs(nanmean(dtptemp));
    %%% timelag -1 to get actual time lag, but account for this later
    [~, maxpeak] = max(dtpxmean);
    maxpeak = maxpeak - 1;
    dtppeakmaxtime(:,1) = maxpeak*10;
    
    
    % construct data table for multilevel regression
    data = table(subj, grouplow, correct, afterbefore, dtppeak(:,1), dtppeak(:,2));
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
    % get before condition in regular performance before group
    aNafter = dafter(dafter.grouplow>0,:);
    bNbefore = dbefore(dbefore.grouplow>0,:);
    
    % run multilevel regressions (roughly equivalent to primary models run in R; note that R is more reliable for precise stats)
    disp('fitting after model')
    lmea = fitglme(aNafter,'acc ~ seqfwd + seqbkw + (1 | subj) + (seqfwd | subj) + (seqbkw | subj)');
    disp('fitting before model')
    lmeb = fitglme(bNbefore,'acc ~ seqfwd + seqbkw + (1 | subj) + (seqfwd | subj) + (seqbkw | subj)');
    
    lmeacoef = dataset2cell(lmea.Coefficients);
    lmebcoef = dataset2cell(lmeb.Coefficients);
    
    % store p-values from after and before models
    pvala(iP,1:2) = [lmeacoef{3,6} lmeacoef{4,6}];
    pvalb(iP,1:2) = [lmebcoef{3,6} lmebcoef{4,6}];
    
    
    %%% get simulated individual difference correlations
    memperf = memperf/max(memperf);
    dtpinddifflosoa = NaN(nSubj,1);
    dtpinddifflosob = NaN(nSubj,1);
    dtpinddiffmaxa = NaN(nSubj,1);
    dtpinddiffmaxb = NaN(nSubj,1);
    for iSj = 1:nSubj
        dtpinddifflosoa(iSj,1) = nanmean(dtpdiffafter(iSj,(dtppeaklosotime(iSj)/10):(dtppeaklosotime(iSj)/10)+2),2);
        dtpinddifflosob(iSj,1) = nanmean(dtpdiffbefore(iSj,(dtppeaklosotime(iSj)/10):(dtppeaklosotime(iSj)/10)+2),2);
        dtpinddiffmaxa(iSj,1) = nanmean(dtpdiffafter(iSj,(dtppeakmaxtime/10):(dtppeakmaxtime/10)+2),2);
        dtpinddiffmaxb(iSj,1) = nanmean(dtpdiffbefore(iSj,(dtppeakmaxtime/10):(dtppeakmaxtime/10)+2),2);
    end
    % loso
    [rafter, pafter] = corr(memperf,dtpinddifflosoa);
    [rbefore, pbefore] = corr(memperf,dtpinddifflosob);
    corrinddifflosoa(iP,1:2) = [rafter pafter];
    corrinddifflosob(iP,1:2) = [rbefore pbefore];
    % max
    [rafter, pafter] = corr(memperf,dtpinddiffmaxa);
    [rbefore, pbefore] = corr(memperf,dtpinddiffmaxb);
    corrinddiffmaxa(iP,1:2) = [rafter pafter];
    corrinddiffmaxb(iP,1:2) = [rbefore pbefore];
    
    
    % optional:  store selected per-subject peak lag and mean fwd-bkw difference
    simpeaklosotime(iP,:) = dtppeaklosotime';
    simpeakmaxtime(iP,1) = dtppeakmaxtime(1);
    dtpdiffaftersim(iP,:,:) = dtpdiffafter;
    dtpdiffbeforesim(iP,:,:) = dtpdiffbefore;
    
    % optional:  save per iteration
    save(['simrandall_ac65_tstart' num2str((tpstart-1)*10) 'ms_p1p' num2str(iP) '_rand' num2str(round(rand,2)*100)],'pvala','pvalb','simpeaklosotime','simpeakmaxtime','dtpdiffaftersim','dtpdiffbeforesim','corrinddifflosoa','corrinddifflosob','corrinddiffmaxa','corrinddiffmaxb','memperf');
end


toc

save(['simall_full_ac65_tstart' num2str((tpstart-1)*10) 'ms_rand' num2str(round(rand,2)*100)],'pvala','pvalb','simpeaklosotime','simpeakmaxtime','dtpdiffaftersim','dtpdiffbeforesim','corrinddifflosoa','corrinddifflosob','corrinddiffmaxa','corrinddiffmaxb','memperf');

