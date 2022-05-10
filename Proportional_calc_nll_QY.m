function single_nll = Proportional_calc_nll_QY(Theta,data)
% calculates the negative log likelihood when fit data with the propor_VP
% model with theta [Jbar, tau]

% Qingqing Yang, qy775@nyu.edu;

if nargin<=0;
    Theta = [6, 0.5]; % debug
    load('exp1_cleandata.mat')
    subjnum = 5;                    % subject number
    data = data{subjnum};  
elseif nargin <=1;
    load('exp1_cleandata.mat')
    subjnum = 5;                    % subject number
    data = data{subjnum}; 
end

% exponentiating appropriate parameters
logflag = [1 1];
logflag = logical(logflag); 


Theta(logflag) = exp(Theta(logflag));

Jbar_total = Theta(1);
Jbar = Jbar_total*1;
tau = Theta(2);
nLL=0;

% get data
data_distance = data{1}(:,1);
nTrials = length(data_distance);

% p(J|Jbar,tau)
nvars = ceil(length(Theta)/2);
JVec = cell(1,nvars);
nJSamp = 500;
JVec = linspace(1e-5,10*Jbar,nJSamp); % get 1 by nJSamp possible JVec
nJs = length(JVec);
Jpdf = gampdf(JVec,Jbar/tau,tau); % get p(J|Jbar,tau), 1 by nJs double
Jpdf (Jpdf==0)=1e-10; % set to arbitrarily small value if zero


% p(Shat|S,J)
%   Y = MVNPDF(X,MU,SIGMA) returns the density of the multivariate normal
%   distribution with mean MU and covariance SIGMA, evaluated at each row
%   of X.
% calculate the p of each trial (nTrials) data given Jbar and sd (nJs)
Sigma = zeros(1,1,nJs*nTrials); % 1 by 2 by nJs*nTrials double
% sd for each trial equals sqrt(1./JVec)
Sigma(1,:,:) = sort(repmat(sqrt(1./JVec(:)),nTrials,1),'descend')'; % sigmas in descending order --> J in ascending order
% for each J, nTrials of data and 0, so repeat sd it nJs, get nTrials*nJs by 2
% where 1st column is the data distance (iterate nJs times), 2nd column is 0.
X= repmat([data_distance(:)],nJs,1);
p_Shat = mvnpdf(X,0,Sigma); %nJs*nTrialsreturn by 1, p_Shat of the data for each 
p_Shat = reshape(p_Shat,nJs,nTrials); % nJs x nTrials
p_Shat(p_Shat == 0) = 1e-10; % set to arbitrarily small value if zero

% \int p(Shat|S,J) p(J) dJ
% Jpdf' nJs by 1 double
% bsxfun(@times,p_Shat,Jpdf') is p(Shat|S,J) p(J)
int=bsxfun(@times,p_Shat,Jpdf'); %nJs by nTrials
pTrials = sum(int); % 1 x nTrials
%LL is sum of (log pTrials) for each trial
nLL = nLL - sum(log(pTrials));

% output
single_nll = nLL;

end % end of nll calc function