%% Simulation and Modeling Practice for VP Models

% Created by Qingqing Yang, qy775@nyu.edu;

% The purpose of this script is to practice simulation process, and
% evaluate the Variable precision models build by Yoo et al. (2018), based
% on the result of simulation.

% In this script, i will do following explorations:
% (1) Simulate a set of data, from evenly allocation VP model parameters.
%   Additionally, compare the data simulated from diff evenly allocation 
%   VP models params combinations in the same experiment.
% (2) Simulate a set of data, from proportional allocation VP model
%   parameters.
% (3) Fit real data with negative Log Likelihood.
% (4) Recover the parameter combination from the simulated data


%% ====================================================================
%  (1)Simulate the experiment data from exp and evenly allocation model 
% =====================================================================
%% Settings of experiment
% params of experiment
nTrials= 100;
nItems = 4;
item_rad = 10;         
item_locs = [1 1; -1 1; -1 -1; 1 -1];
item_locs = item_locs.*item_rad;
rng('default');

tar_quadrant=[1;2;3;4];
tar_quadrant=repmat([1;2;3;4],[nTrials/nItems,1]);
tar_quadrant=Shuffle(tar_quadrant);
tar_loc=[]; % I'll set the exact tar_loc during simulated experiment
% Since the tar_loc's arrangement for each subject is different

%% Settings of subjects' precision gamma distribution
    % For simplest situation, let's assume that the when subject trying to
    % remember the stimuli locations,they tend to allocate same amount of
    % resources, to each memory item, even with different priority. 
    % Therefore the memory of each target should be drawn from a 2D normal 
    % distribution with a mean of that specific target location, and a 
    % varying standard deviation from the same distribution.
   
    % Note that the sd changes in each trial. To explain: if we assume that
    % the sd on x and y axis are the same for each trial, for
    % simplification, based on the hypothesis in VP Model, the sd for each 
    % trial should be 1/sqrt(J). Additionally, J follow a gamma
    % distribution with mean parameter Jbar, and scale parameter tau. The 
    % distribution of J is unique for each target with different priority, 
    % for each different subject.
    % Therefore, for each participant, each target, we could gain One J for
    % each trial from a gamma distribution with mean Jbar and shape variable
    % tau, and then get the sd of memory in each axis for each trial.

% params of a single subject memory precision (model)
% Jbar as mean, tau as scale parameter, so Jbar/tau is the shape parameter
% of gamma distribution.
% here we test 3 groups of para combinations, or say 3 subjects
Jbar_1 = 3*randn()+randi(20);
tau_1 = randn()+randi(5);
Jbar_total_1=Jbar_1*nItems;
Jbar_2 = 3*randn()+randi(20);
tau_2 = randn()+randi(5);
Jbar_total_2=Jbar_2*nItems;
Jbar_3 = 3*randn()+randi(20);
tau_3 = randn()+randi(5);
Jbar_total_3=Jbar_3*nItems;

Jbar=[Jbar_1 Jbar_2 Jbar_3];
tau=[tau_1 tau_2 tau_3];

sd=[];
memories=[];
relative_me=[];
error = [];

% plot gamma distribution of this participant memory precision
xx = linspace(0,40,100);
colorMat = [1 0 0; 0 1 0; 0 0 1];
figure; hold on
for i =1:3
blah = gampdf(xx,Jbar(i)/tau(i),tau(i));
plot(xx,blah./sum(blah),'Color',colorMat(i,:));
legend('Jbar=Jbar(i); tau=tau(i)')
end 
legend('Jbar_1, tau_1','Jbar_2, tau_2','Jbar_3, tau_3' )

%% Simulated memory
% For each param combination, or say subject, 
% presenting Target, and simulate memory, calculate the error
iitem=1;
sd=[];
memories=[];
relative_me=[];
error = [];

for i =1:3
% Based on specific para combinations, calculate the J for each trial
% shape_param=Jbar/tau;
J = gamrnd(Jbar(i)/tau(i),tau(i),[nTrials,1]);
% convert into sigma for gaussian distribution
sd = 1./sqrt(J);

    for iitem = 1:nTrials;    
    % the location that target appear
    tar_loc(iitem,:) = item_locs(tar_quadrant(iitem),:);
    % the memory location
    memories(iitem,2*i-1)= tar_loc(iitem,1)+ sd(iitem).*randn(1,1);
    memories(iitem,2*i)= tar_loc(iitem,2)+ sd(iitem).*randn(1,1);
    % the relative memory (error)location based on target location
    relative_me(iitem,2*i-1)=memories(iitem,2*i-1)-tar_loc(iitem,1);
    relative_me(iitem,2*i)=memories(iitem,2*i)-tar_loc(iitem,2);
    % calculate the euclidean distance of memory error
    error(iitem,i) = sqrt(relative_me(iitem,2*i-1).^2 +relative_me(iitem,2*i).^2);
    end   

end

% For each para combination, plot the relative memory location, which 
% could be also consided as directional memory errors
figure;
for i = 1:3
subplot(1,3,i);
plot(0,0,'r.','MarkerSize',24); hold on;
plot(relative_me(:,2*i-1),relative_me(:,2*i),'k.')
axis equal
axis([-2 2 -2 2])
end
% legend('Jbar_1, tau_1','Jbar_2, tau_2','Jbar_3, tau_3' )

figure; 
for i = 1:3
subplot(3,1,i);
histogram(error(:,i));
xlabel('Euclidean distance as memory error (rad)')
ylabel('frequency')
end


%% ====================================================================
%  (2)Simulate a set of data, from proportional allocation VP model params.
% =====================================================================
%% Based on priority, set the gamma distributions
% The allocation of resource in proportional allocation, is equal to target
% behavioral relevances, or say priorities.
% so here in Yoo et al. (2018) the priorities are [0.6 0.3 0.1] for three
% possible targets, we also set the allocation as the same.

% model parameters
Jbar_total = 10;

% % when fixed tau, changing Jbar
% taus = [0.5 0.5 0.5]
% allocationVec = [0.6 0.3 0.1]; 

% when fixed the priority, changing taus
taus = [2 0.8 0.3];
allocationVec = [0.33 0.33 0.33]; 

% draw the distributions
xx = linspace(0,10,100);
colorMat = [1 0 0; 0 0 1; 0 0 0];
figure; hold on
for ipriority = 1:length(allocationVec)
    Jbar = Jbar_total*allocationVec(ipriority);
    
    blah = gampdf(xx,Jbar/taus(ipriority),taus(ipriority));
    plot(xx,blah./sum(blah),'Color',colorMat(ipriority,:))
end
xlabel('precision (J)')
ylabel('proportion')

%% Experiment settings
nItems = 4;
item_rad = 5;         
item_locs = [1 1; -1 1; -1 -1; 1 -1];
item_locs = item_locs.*item_rad;

% % for fixed tau, varying Jbar simulation
% nTrials = [260 160 80];
% nTrialsTotal = 500;

% for fixed Jbar, varying tau simulation
nTrials = [160 160 160];
nTrialsTotal = 480;

rng('default');

tar_quadrant=[1;2;3;4];

tar_loc=[]; % I'll set the exact tar_loc during simulated experiment
% Since the tar_loc's arrangement for each subject is different
%% Simulate the memory data
model = 'proportional';
expnumber = 1; %1 (no disc). 2 (with disc);
Theta(1)= Jbar_total;
expPriorityVec = allocationVec;
nPriorities = length(expPriorityVec);
memories = cell(1,nPriorities);
relative_me=[];
error = cell(1)
for ipriority = 1:nPriorities    % for each item...
    tau=taus(ipriority);
    Theta(2)= tau;
    Theta=[Jbar_total tau];
    priority = expPriorityVec(ipriority);      % item priority
    p = allocationVec(ipriority);             % proportion allocated to item
    Jbar = Jbar_total*p;          % item precision
    N = nTrials(ipriority);             % number of trials for current item
    J = gamrnd(Jbar/tau,tau,[N,1]); % compute the J for each priority
    sd = 1./sqrt(J); % convert into sigma for gaussian distribution
    tar_quadrant=repmat([1;2;3;4],[N/nItems,1]);
    tar_quadrant=Shuffle(tar_quadrant);
    
    for iTrial = 1: N
        % target location
        tar_loc = item_locs(tar_quadrant(iTrial),:);      % item location
        % generate memory
        % the memory location
        memories {1, ipriority}(iTrial,2*ipriority-1)= tar_loc(1,1)+ sd(iTrial).*randn(1,1);
        memories{1, ipriority}(iTrial,2*ipriority)= tar_loc(1,2)+ sd(iTrial).*randn(1,1);
        % the relative memory (error)location based on target location
        relative_me(iTrial,2*ipriority-1)= memories{1, ipriority}(iTrial,2*ipriority-1)-tar_loc(1,1);
        relative_me(iTrial,2*ipriority)= memories{1, ipriority}(iTrial,2*ipriority)-tar_loc(1,2);
        % calculate the euclidean distance of memory error
        error{1,1}{1, ipriority}(iTrial,1) = sqrt(relative_me(iTrial,2*ipriority-1).^2 +relative_me(iTrial,2*ipriority).^2);
    end
end

% For each para combination, plot the relative memory location, which 
% could be also consided as directional memory errors
figure;
for i = 1:3
subplot(1,3,i);
plot(0,0,'r.','MarkerSize',40); hold on;
% plot(relative_me(:,2*i-1),relative_me(:,2*i),'ko')
s=scatter(relative_me(:,2*i-1),relative_me(:,2*i),'filled');
distfromzero = sqrt(relative_me(:,2*i-1).^2 + relative_me(:,2*i).^2);
s.AlphaData = distfromzero;
% s.MarkerFaceAlpha = 'flat';
s.MarkerFaceAlpha = 0.15;
s.MarkerFaceColor = [0 0 1];
axis equal
axis([-3.5 3.5 -3.5 3.5])
end
% legend('Jbar_1, tau_1','Jbar_2, tau_2','Jbar_3, tau_3' )

figure; 
for i = 1:3
subplot(3,1,i);
histogram(error{1}{1,i});
xlabel('Euclidean distance as memory error (rad)')
ylabel('frequency')
xlim([0,2.5])
end

%% For the time sake, for simulation, we could also use the simulate_data.m by Yoo et al. (2018)
% which could be found in github.com/aspenyoo/WM_resource_allocation
clear memories, 
clear error;
clear Theta;
clear expPriorityVec;
expnumber = 1;
Theta=[10, 0.5];
nTrials = [260 160 80];
expPriorityVec = [0.6 0.3 0.1];

error = simulate_data(model,expnumber,Theta,nTrials,expPriorityVec);

% plot the memory errors
nPriorities = length(expPriorityVec); 
xlims = linspace(0,5,16); % x values for histogram
figure; hold on;
for ipriority = 1:nPriorities
    datacounts = hist(error{1}{ipriority},xlims);
    plot(xlims,datacounts./sum(datacounts),'Color',colorMat(ipriority,:));
end


%% ====================================================================
%  (3)Fit real data with negative Log Likelihood
% =====================================================================

% First, practice to calculate a negative log likelihood
% load one subject's data from Yoo et al. (2018) exp1
load('exp1_cleandata.mat')
subjnum = 5;                    % subject number
data = data{subjnum};   

model = 'proportional';
Jbar_total = 10;
tau = 0.5;
Theta = [Jbar_total tau];
expPriorityVec = [0.6 0.3 0.1];
fixparams = [];

%% calculate -LL with calc_nLL.m by Yoo et al. (2018)
nLL = calc_nLL(model,Theta,data,expPriorityVec,fixparams);
% calc_nLL calculates negative LL of parameters given data and model
% % p(J|Jbar,tau)
% % p(Shat|S,J)
% % \integral p(Shat|S,J) p(J) dJ
% nLL = 0;
% nLL = nLL - sum(log(pTrials));

% I wrote my own script for the data that contains only 1 target, while
% here the data contains error data for 3 target. here, data is a 1 by 3
% cell, with data{1,1}, data{1,2}, data{1,3} respectively represent data
% for one target

theta = [Jbar_total*expPriorityVec(1), 0.5];
single_nll=Proportional_calc_nll_QY(theta,data) 
% this is not correct, value is not the same, something is off

expPriorityVec=[1];
nLL2 = calc_single_nLL('proportional',theta,data,expPriorityVec)
% I revised the calc_nLL to get calc_single_nLL, which only takes the
% data{1,1} in and calculate the nLL.
%% model fitting
model = 'proportional';             % model name
load('exp1_cleandata.mat');
subjnum = 5;                    % subject number
data = data{subjnum};
exppriorityVec = [0.6 0.3 0.1];            % experimental priority vector
runlist = 1;                    % ignore. which idxs of total runs for current model/data
runmax = 20;                    % ignore. number of runs per model/data
fixparams = [];                 % fixed parameters, ignore for now

% fit parameter
[ML_parameters, nLLVec] = fit_parameters(model,data,exppriorityVec,runlist,runmax,fixparams)
% fit_parameters.m is from Yoo et al. (2018)
run=1;
[ML_parameters2, nLLVec,runlist_completed]=Proportional_fitparams_QY(data,run)
%% plotting model fits

% plot data
xlims = linspace(0,10,16); % x values for histogram
figure; hold on;
for ipriority = 1:nPriorities
    datacounts = hist(data{ipriority},xlims);
    plot(xlims,datacounts./sum(datacounts),'Color',colorMat(ipriority,:));
end

% get model prediction
expnumber = 1;
error = simulate_data(model,expnumber,ML_parameters,nTrials,expPriorityVec);
error = error{1};

% plot model prediction with dotted lines
for ipriority = 1:nPriorities
    datacounts = hist(error{ipriority},xlims);
    plot(xlims,datacounts./sum(datacounts),':','Color',colorMat(ipriority,:));
end


%% ====================================================================
%  (4)Recover the parameters combination from the simulated data
% =====================================================================
%% simulate by myself
% simulate data with proportional VP model
theta = [3, 0.5];
nTrials = 200;
single_error=Proportional_VP_single_simulator_QY(theta,nTrials);
% return a simulated single_error{1,1} cell which is nTrials by 1 

%% calculate the nLL values by myself
theta = [3, 0.5];
 single_nll=Proportional_calc_nll_QY(theta,single_error) % this is not correct
% something is off
% nll keep getting smaller as Jbar gets smaller...

expPriorityVec=[1];
 nLL = calc_single_nLL('proportional',theta,single_error,expPriorityVec)
%% recover the params by myself
run = 1; %number of optimizations for a given Model and Data.
data = single_error;
[ML_parameters,nLLVec,runlist_completed]=Proportional_fitparams_QY(data,run);
ML_parameters=log(ML_parameters)

% something is wrong when i change the old fit_parameters.m, cause the
% theta for calc_single_nLL is originally Jbartotal instead of Jbar... 

% because Proportional_calc_nll_QY is not correct, this is not correct at
% first when i am using Proportional_calc_nll_QY.m

% now i used calc_single_nLL.m inside Proportional_fitparams_QY.m, which
% works better, but still something is off.

%% draw a correlation matrix 
% it takes really a long time to run
nTrials = 200;
run = 1;
maxJbar=3;
maxtau=1;
nJbar =10;
ntau=5;
Jbarval=linspace(1,maxJbar,nJbar);
tauval=linspace(1e-3,maxtau,ntau);
theta=[];
theta_recover=[];
i = 1;
iJbar=1;
itau=1;
for iJbar=1:length(Jbarval)
    for itau=1:length(tauval)
        theta(i,:) = [Jbarval(iJbar),tauval(itau)];
        single_error=Proportional_VP_single_simulator_QY(theta(i,:),nTrials);        
        [ML_parameters, nLLVec,runlist_completed]=Proportional_fitparams_QY(single_error,run);
         ML_parameters=log(ML_parameters);
        theta_recover(i,:)=ML_parameters;
        i=i+1;
    end
end

%% plot the theta and theta_recover
% xlims = linspace(0,maxJbar,20); % x values for histogram
figure; hold on;
% theta_recover=real(theta_recover);
plot(theta_recover(:,1),theta_recover(:,2),'k.'); hold on;
plot(theta(:,1),theta(:,2),'ko');  
xlabel('Jbar')
ylabel('tau')
axis([1 3 0 1]);

%% since it looks weird and took long time to recover the param, save it.
filename='./qy_modelling results/theta&theta2.mat';
save(filename,'theta','theta_recover');

