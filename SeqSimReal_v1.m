% seqSimReal.m
% 
% G. Elliott Wimmer, Yunzhe Liu 2020
% 
% permute data labels to compute false positive rate for sequenceness
% relationship to behavior given real sequenceness input


%% initial set up
clear



% set variables
maxLag =        31; % 31 = 350 ms lag

% set number of simulations
nsim =       10000;
startsim =       1;
% load in intermediate file if starting from >1
if startsim>1
    simtoload = ['simrealall_corrrand_p1p' num2str(startsim-1)];
    load(simtoload);
    fprintf(['\n' 'Loaded ' simtoload '\n\n'])
end

fprintf(['\n' 'Running ' num2str(nsim) ' simulations.' '\n\n'])

fprintf(['\n' 'Starting at ' num2str(startsim) '.' '\n\n'])


%% set correct/incorrect permutation
% 1 = permuting correct labeling / 0 = preserving labeling
setcorrperm = 1;
if setcorrperm==1
    fprintf('Randomizing correct/incorrect assignment!\n\n');
else
    fprintf('NOT randomizing corr/incorr.\n\n');
end


%% load behavior and sequenceness
load seq_all_xp12_nocon_sfsb_n25 % nsubj*ntrials length matrices of sequenceness values (40ms-360ms lag). sf, sb raw; sfmeancorr, sbmeancorr mean-corrected by column for regressions
load seq_all_behavior % nsubj*ntrials length matrices of behavior
load seq_all_inddiff % nsubj length matrices of mean behavior and group membership
nSubj = length(subjlist);


input('\nConfirm the above\n')
fprintf('Ok...\n\n');



% loop through permutations
for iP = startsim:nsim
    
    tic
    
    fprintf(['\n' 'perm = ' num2str(iP) '\n\n'])
    
    %% randomize correct assignment
    correctall = []; % correct
    for iSj=1:nSubj
        
        % randomize correct
        correct = correct_allsubj{iSj};
        if setcorrperm==1
            correct(:,2) = rand(length(correct),1);
            correct = sortrows(correct,2);
            correct = correct(:,1);
        end
        % concat
        correctall = [correctall; correct];
    end
    
    % get difference of fwd-bkw
    sdiffall = sf-sb;
    sdiff0all = sfmeancorr-sbmeancorr;
    % set storage
    dtp0diffafter = NaN(nSubj,maxLag+1);
    dtp0diffbefore = NaN(nSubj,maxLag+1);
    dtpdiffafter = NaN(nSubj,maxLag+1);
    dtpdiffbefore = NaN(nSubj,maxLag+1);
    
    % get data for calculating peak after condition correct trial replay
    colsubj = 1;
    colafterbefore = 4;
    afterbeforeall = dataall(:,colafterbefore);
    subjall = dataall(:,colsubj);
    
    for iSj = 1:nSubj
        
        subjtrials = find(subjall==subjlist(iSj,:));
        corrsubj = correctall(subjtrials);
        afterbeforesubj = afterbeforeall(subjtrials);
        sdiff0subj = sdiff0all(subjtrials,:);
        sdiffsubj = sdiffall(subjtrials,:);
        
        dtp0diffafter(iSj,:) = nanmean(sdiff0subj(corrsubj==1 & afterbeforesubj==1,:));
        dtp0diffbefore(iSj,:) = nanmean(sdiff0subj(corrsubj==1 & afterbeforesubj==-1,:));
        
        dtpdiffafter(iSj,:) = nanmean(sdiffsubj(corrsubj==1 & afterbeforesubj==1,:));
        dtpdiffbefore(iSj,:) = nanmean(sdiffsubj(corrsubj==1 & afterbeforesubj==-1,:));
    end
    
    %% get leave-one-subject-out (LOSO) cross-validation peak timepoint per subject
    maxtime = 31; % 31 = 350 ms lag
    tpstart = 1; % 1 = 40 ms lag
    tlagbuffer = 4;
    dtpxmean = NaN(nSubj,maxtime);
    dtppeaktime = NaN(nSubj,1);
    dtppeak = [];
    
    for iSj = 1:nSubj
        
        dtptemp = dtpdiffafter(:,1:maxtime);
        dtptemp(iSj,:) = NaN;
        dtpxmean(iSj,:) = abs(nanmean(dtptemp));
        
        % timelag -1 to get actual time lag, but account for this later
        subjpeak = (find(dtpxmean(iSj,:)==max(dtpxmean(iSj,tpstart:maxtime)))-1);
        
        dtppeaktime(iSj,1) = subjpeak(1)*10;
        
        if subjpeak==0
            subjlagrange = subjpeak+1:subjpeak+2;
        else
            subjlagrange = subjpeak:subjpeak+2;
        end
        peakfwd = nanmean(sfmeancorr(subjall==subjlist(iSj),subjlagrange),2);
        peakbkw = nanmean(sbmeancorr(subjall==subjlist(iSj),subjlagrange),2);
        
        dtppeak = [dtppeak; [peakfwd peakbkw]];
    end
    
    %% for individual differences correlations, get absoluate value max sequenceness lag
    dtptemp = dtpdiffafter(:,1:maxtime);
    dtptemp(:,1:tpstart-1) = NaN;
    dtpxmean = abs(nanmean(dtptemp));
    
    % store maxtime timelag -1 to get actual time lag, but account for this later
    [~, maxpeak] = max(dtpxmean);
    maxpeak = maxpeak - 1;
    dtppeakmaxtime(:,1) = maxpeak*10;
    
    %% create data table for fitglme
    data = array2table([dataall dtppeak correctall]); % concat behavior, seq peak, and (permuted) correct label
    data.Properties.VariableNames{1} = 'subj';
    data.Properties.VariableNames{2} = 'groupafter';
    data.Properties.VariableNames{3} = 'groupbefore';
    data.Properties.VariableNames{4} = 'afterbefore';
    data.Properties.VariableNames{5} = 'corrm1x';
    data.Properties.VariableNames{6} = 'corrm2x';
    data.Properties.VariableNames{7} = 'corrp1x';
    data.Properties.VariableNames{8} = 'corrp2x';
    data.Properties.VariableNames{9} = 'seqfwd';
    data.Properties.VariableNames{10} = 'seqbkw';
    data.Properties.VariableNames{11} = 'acc';
    
    % subset data into n=17 regular-performers in after condition; n=18 regular-performers in before condition
    dafter = data(data.afterbefore==1,:);
    dbefore = data(data.afterbefore==-1,:);
    
    aNafter = dafter(dafter.groupafter>0,:);
    bNbefore = dbefore(dbefore.groupbefore>0,:);
    
    % get dat for inddiff regressions
    dtcorrinddifflosoa = NaN(nSubj,1);
    dtcorrinddifflosob = NaN(nSubj,1);
    dtcorrinddiffmaxa = NaN(nSubj,1);
    dtcorrinddiffmaxb = NaN(nSubj,1);
    for iSj = 1:nSubj
        dtcorrinddifflosoa(iSj,1) = nanmean(dtpdiffafter(iSj,(dtppeaktime(iSj)/10):(dtppeaktime(iSj)/10)+2),2);
        dtcorrinddifflosob(iSj,1) = nanmean(dtpdiffbefore(iSj,(dtppeaktime(iSj)/10):(dtppeaktime(iSj)/10)+2),2);
        dtcorrinddiffmaxa(iSj,1) = nanmean(dtpdiffafter(iSj,(dtppeakmaxtime/10):(dtppeakmaxtime/10)+2),2);
        dtcorrinddiffmaxb(iSj,1) = nanmean(dtpdiffbefore(iSj,(dtppeakmaxtime/10):(dtppeakmaxtime/10)+2),2);
    end
    
    %% run inddiff regressions
    [rafter, pafter] = corrcoef(n25_perfafter,dtcorrinddifflosoa);
    [rbefore, pbefore] = corrcoef(n25_perfbefore,dtcorrinddifflosob);
    corrinddifflosoa(iP,1:2) = [rafter(2) pafter(2)];
    corrinddifflosob(iP,1:2) = [rbefore(2) pbefore(2)];
    
    [rafter, pafter] = corrcoef(n25_perfafter,dtcorrinddiffmaxa);
    [rbefore, pbefore] = corrcoef(n25_perfbefore,dtcorrinddiffmaxb);
    corrinddiffmaxa(iP,1:2) = [rafter(2) pafter(2)];
    corrinddiffmaxb(iP,1:2) = [rbefore(2) pbefore(2)];
    
    
    %% run multilevel regressions
    fprintf('fitting after model...\n')
    % optional full model from manuscript; results equivalent but processing demands significantly greater
    %lmea = fitglme(aNafter,'acc ~ seqfwd + seqbkw + corrm1x + corrm2x + corrp1x + corrp2x + (1 | subj) + (seqfwd | subj) + (seqbkw | subj) + (corrm1x | subj) + (corrm2x | subj) + (corrp1x | subj) + (corrp2x | subj)');
    %rowstat = 7;
    rowstat =    3;
    colpval =    6;
    coltstat =   4;
    
    lmea = fitglme(aNafter,'acc ~ seqfwd + seqbkw + (1 | subj) + (seqfwd | subj) + (seqbkw | subj)');
    fprintf('fitting before model...\n\n\n')
    lmeb = fitglme(bNbefore,'acc ~ seqfwd + seqbkw + (1 | subj) + (seqfwd | subj) + (seqbkw | subj)');
    
    % store p-values and t-stats from after and before models
    lmeacoef = dataset2cell(lmea.Coefficients);
    lmebcoef = dataset2cell(lmeb.Coefficients);
    
    pvala(iP,1:2) = [lmeacoef{rowstat,colpval} lmeacoef{rowstat+1,colpval}];
    pvalb(iP,1:2) = [lmebcoef{rowstat,colpval} lmebcoef{rowstat+1,colpval}];
    
    tstata(iP,1:2) = [lmeacoef{rowstat,coltstat} lmeacoef{rowstat+1,coltstat}];
    tstatb(iP,1:2) = [lmebcoef{rowstat,coltstat} lmebcoef{rowstat+1,coltstat}];
    
    % store peak lag selection and mean fwd-bkw difference
    simpeaktime(iP,:) = dtppeaktime+(tlagbuffer*10)';
    
    %% optional:  save per iteration / after every n iterations
    if rem(iP,500)==0
        save(['simrealall_corrrand_p1p' num2str(iP)],'tpstart','nSubj','tstat*','pval*','simpeaktime','corrinddiff*');
    end
    
    toc
end


%% individual differences analyses:  compute false positive rate and adjusted 5% threshold

% empirical permutation-derived adjusted 5% threshold
threshadjinddiffafter = prctile(corrinddiffmaxa(:,2),5);
% 25k:  0.0399
threshadjinddiffbefore = prctile(corrinddiffmaxb(:,2),5);
% 25k:  0.0507

% empirical permutation-derived false positive rate
fposrateinddiffafter = nanmean(corrinddiffmaxa(:,2)<0.05);
% 25k:  0.0582
fposrateinddiffbefore = nanmean(corrinddiffmaxb(:,2)<0.05);
% 25k:  0.0488

% likelihood of finding a difference between after and before correlations >= 0.8
pvalinddiffconj = abs(corrinddiffmaxa(:,1)-corrinddiffmaxb(:,1))>=0.8; % actual data correlation diff ~ 0.8 (data: r>0.40, r>-0.40)
pvalinddiffconj = sum(pvalinddiffconj)./length(pvalinddiffconj);
% 25k:  0.0076



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


%% save full permutation results, adjusted 5% thresholds, and false positive rates
if setcorrperm==0
    savename = ['simrealall_full_corrreal'];
else
    savename = ['simrealall_full_corrrand_p1p' num2str(iP) '_rand' num2str(round(rand,2)*100)];
end
save(savename,'nSubj','tstat*','pval*','simpeaktime','corrinddiff*','thresh*','fpos*');



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
title(['a+b (fwd-bkw) ' num2str(length(corrinddiffmaxa)) ' inddiff max realperm; <5% after ' num2str(round(mean(corrinddiffmaxa(:,2)<0.05),4)) '; before ' num2str(round(mean(corrinddiffmaxb(:,2)<0.05),4))]);
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

figname = (['mem_inddiff_realpermmax_permn' num2str(length(pvala))]);
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
title(['a+b fwd+bkw ' num2str(length(pvala(:,1))) ' realperm; <5% after ' num2str(round(sum(pvala(:)<0.05)/length(pvala(:)),4)) '; before ' num2str(round(sum(pvalb(:)<0.05)/length(pvalb(:)),4))]);
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

figname = (['mem_realpermloso_permn' num2str(length(pvala))]);
saveas(gcf,figname,'fig')
saveas(gcf,figname,'png')