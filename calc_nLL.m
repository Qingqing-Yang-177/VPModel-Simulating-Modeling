function nLL = calc_nLL(model,Theta,data)
% CALC_NLL(JBAR_TOTAL,TAU,BETA)
%
% CALC_NLL: calculates negative log likelihood of parameter combination for
% the optimal model
% 
% ================= INPUT VARIABLES ======================
% MODEL: 1 (optimal) or 2 (not optimal)
% 
% THETA = vector of parameters. For model 1, [ Jbar_total tau beta]. 
%   For model 2, [Jbar_total tau beta p_high p_med]
%       JBAR_TOTAL: total amount of resources across priorities
%       TAU: second parameter of gamma noise distribution
% 
% DATA: 1 x 3 cell each containing nTrials x 2 matrix. 
%   struct organization: high, med, low priority trials
%   1st column: distance between target and final saccade
%   2nd column: radius of disc
% 
% ================= OUTPUT VARIABLES ================
% NLL: -L(Theta|data,model)
%
% -----------------------
%      Aspen H. Yoo
%   aspen.yoo@nyu.edu


% exponentiating appropriate parameters
switch model
    case 1  % optimal model
        logflag = logical([1 1 0]);
    case 2 % not optimal model
        logflag = logical([1 1 0 0 0]);
        
        % the proportion of resource allocation cannot exceed 1
        if sum(Theta(4:5)) > 1
            nLL = Inf;
            return;
        end
end
Theta(logflag) = exp(Theta(logflag));
Jbar_total = Theta(1);
tau = Theta(2);
beta = Theta(3);

% data stuff
priorityVec = [0.6 0.3 0.1];
nPriorities = length(priorityVec);

switch model
    case 1 % optimal
        % calculate the proportions that maximize expected utility
        pVec = calc_optimal_pVec(Theta);
    case 2 % not optimal
        pVec = [Theta(4:5) 1-sum(Theta(4:5))];
        if pVec(3) <= 0 % proportion allocated to lowest priority must be positive
            nLL = Inf;
            return
        end
end

% loading vector of disc radii
[rVec] = loadvar('rVec');
rVec = rVec(:); % vertical

nLL = 0;
for ipriority = 1:nPriorities
    Jbar = Jbar_total*pVec(ipriority); % Jbar for current priority condition
    
    % clear data used in previous priorities
    clear idx1 idx2 data_r_reshaped
    
    % get subject data
    data_distance = data{ipriority}(:,1);
    data_r = data{ipriority}(:,2);
    
    % p(J|Jbar,tau)
    [JVec] = loadvar({'JVec',Jbar,tau}); % values of J
    nJs = length(JVec);
    Jpdf = gampdf(JVec,Jbar/tau,tau); % probability of that J value
    Jpdf = Jpdf./qtrapz(Jpdf); % normalize
%     if any(Jpdf > 1); Jpdf = Jpdf./sum(Jpdf); return; end % weird shit happens at extremely low taus, where the probability of the lowest value is >1.
    
    % p(Shat|S,J)
    nTrials = length(data_distance);
    Sigma = zeros(1,2,nJs*nTrials);
    Sigma(1,:,:) = sort(repmat(sqrt(1./JVec(:)),nTrials,2),'descend')'; % sigmas in descending order --> J in ascending order
    p_Shat = mvnpdf(repmat([data_distance(:) zeros(nTrials,1)],nJs,1),0,Sigma);
    p_Shat = reshape(p_Shat,nTrials,nJs)'; % nJs x nTrials
    
    % p(rVec|J,beta) (a range of r to get entire probability dist)
    pdf_r = calc_pdf_r(beta, JVec); % rVec x JVec
    
    % calculate p(r|J,beta) (get indices of which r in rVec is closest to actual r)
    data_r = data_r(:)';  % horizontal vector
    firstidxs = bsxfun(@(x,y) x == x(find((x-y)<=0,1,'last')),rVec,data_r);
    lastidxs = bsxfun(@(x,y) x == x(find((x-y)>=0,1,'first')),rVec,data_r);
    idx1(:,1,:) = firstidxs;
    idx2(:,1,:) = lastidxs;
    
    xdiff = diff(rVec(1:2));
    ydiff = sum(bsxfun(@times,pdf_r,idx2)) - sum(bsxfun(@times,pdf_r,idx1)); % 1 x JVec x nTrials
    slope = ydiff./xdiff; % 1 x JVec x nTrials
    
    % linearly interpolate
    data_r_reshaped(1,:,:) = data_r; % 1 x 1 x nTrials
    p_r = squeeze(bsxfun(@times,slope,bsxfun(@minus,data_r_reshaped,sum(bsxfun(@times,rVec,idx1)))) + sum(bsxfun(@times,pdf_r,idx1))); % JVec x nTrials
    
    % \int p(Shat|S,J) p(r|J) p(J) dJ
    pTrials = sum(bsxfun(@times,p_Shat.*p_r,Jpdf')); % 1 x nTrials
    nLL = nLL - sum(log(pTrials));
    
    
    
    
%     % p(Shat|S) = \int p(Shat|S,J) p(J) dJ
%     p_Shat = qtrapz(bsxfun(@times,p_Shat,Jpdf),2);
%     
%     % get pdf of p(r|J,beta)
%     pdf_r = calc_pdf_r(beta, JVec); % rVec x JVec
%     pdf_r = bsxfunandsum(@times,pdf_r,Jpdf,2); % rVec x 1
%     
%     % p(r): probability of responding r given the parameters
%     data_r = data_r(:)';  % make sure it is horizontal vector
%     firstidxs = bsxfun(@(x,y) x == x(find((x-y)<=0,1,'last')),rVec,data_r);
%     lastidxs = bsxfun(@(x,y) x == x(find((x-y)>=0,1,'first')),rVec,data_r);
%     
%     % linearly interpolate
%     slope = (sum(bsxfun(@times,pdf_r,lastidxs)) - sum(bsxfun(@times,pdf_r,firstidxs)))./diff(rVec(1:2));
%     p_r = slope.*(data_r - sum(bsxfun(@times,rVec,firstidxs))) + sum(bsxfun(@times,pdf_r,firstidxs));
%     
%     % nLL
%     nLL = nLL -sum(log(p_Shat)) -sum(log(p_r));
%     %     nLL = nLL -sum(log((1-lapse).*p_Shat + lapse.*p_Shat_lapse)) -sum(log(p_r));
end
