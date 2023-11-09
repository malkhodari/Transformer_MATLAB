function [q, k, v, order] = iOrderForBuiltin(q, k, v, channelDim, obsDim, positionDim)
order = [channelDim, positionDim, obsDim];
% Manage singleton dimensions to obtain a valid permutation vector
if ~any(order==2)    
    order = [order 2];
end
if ~any(order==1)
    order = [order 1];
end
q = permute(q, order);
k = permute(k, order);
v = permute(v, order);
end