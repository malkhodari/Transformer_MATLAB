function [X, dropoutMask] = iDropout(X, prob)
if prob == 0
    dropoutMask = cast(1, 'like', X);
else
    dropoutScaleFactor = cast(1 - prob, 'like', X);
    dropoutMask = (rand(size(X), 'like', X) > prob) / dropoutScaleFactor;
    X = X.*dropoutMask;
end
end