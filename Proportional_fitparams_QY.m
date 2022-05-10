function [ML_parameters, nLLVec,runlist_completed] = Proportional_fitparams_QY(data,run)
% Get the best fitting params with its nLL value, when fitting data to the
% Proportional VP model

if nargin<=0;
    load('exp1_cleandata.mat')
    subjnum = 5;                    % subject number
    data = data{subjnum}; 
end

% lower and upper bounds, logflags
% Jbar_total, tau search from [1e-5 1e-3] to [50 10];
lb = [1e-5 1e-3]; 
ub = [50 10];
plb = [0.5 0.01];
pub = [10 1];
logflag = [1 1];
nParams = length(logflag);
x0_list = [];

% now we used the latin hypercube
while size(x0_list,1) < run
    rng(0);
    x0_list = lhs(run,nParams,plb,pub,[],1e3); % optimize the lhs up to 1e3 times
   %LHS(N,P) creates a Latin hypercube sample design with N points on the unit hypercube [0,1]^P.
end

% optimize for starting values
ML_parameters = [];
nLLVec = [];
runlist_completed = [];
runlist = [];
for irun = 1:run
    runlist(irun)=irun;   
    rng(runlist(irun));
    x0 = x0_list(runlist(irun),:);
    fun = @(x) Proportional_calc_nll_QY(x,data);
    x = bads(fun,x0,lb,ub,plb,pub); 
%   X = BADS(FUN,X0,LB,UB,PLB,PUB) starts at X0 and finds a local minimum X
%   to the function FUN. It also specifies a set of plausible lower and
%   upper bounds such that LB <= PLB <= X0 <= PUB <= UB. Both PLB and PUB
%   are used to design the initial mesh of the direct search, and represent 
%   a plausible range for the optimization variables.

    fval = fun(x); % nLL for local minima
    x(logflag) = exp(x(logflag));
    bfp = x;
    ML_parameters = [ML_parameters; bfp];
    nLLVec = [nLLVec fval];
    runlist_completed = [runlist_completed runlist(irun)];
end %end of irun

end