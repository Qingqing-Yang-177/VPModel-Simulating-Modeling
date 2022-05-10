function error_simulated=Proportional_VP_single_simulator_QY(Theta,nTrials)
% Theta=[Jbar tau];
% allocationVec is the priorities [] for targets
% model = 'proportional';

if nargin<=0;
    Theta = [3, 0.5]; % debug
    nTrials = 200;
elseif nargin <=1;
    nTrials = 200;
end

% gamma distribution
Jbar = Theta(1);
tau = Theta(2);
N = nTrials;             % number of trials for current item
J = gamrnd(Jbar/tau,tau,[N,1]); % compute the J for each priority
sd = 1./sqrt(J); % convert into sigma for gaussian distribution

%simulate
relative_me=[];
error=cell(1);
for iTrial = 1: N
    % the relative memory (error)location based on target location
    relative_me(iTrial,1)= sd(iTrial).*randn(1,1);
    relative_me(iTrial,2)= sd(iTrial).*randn(1,1);
    % calculate the euclidean distance of memory error
    error{1,1}{1, 1}(iTrial,1) = sqrt(relative_me(iTrial,1).^2 +relative_me(iTrial,2).^2);
end
error_simulated=error{1}

% plot the simulated data
figure;
xlims = linspace(0,10,18);
datacounts = hist(error_simulated{1},xlims);
plot(xlims,datacounts./sum(datacounts));

end % end of proportaionl simulator