function [labelDim, hasLabelDim] = iFindDimension(labelsData, label)
labelDim = find(labelsData == label);
hasLabelDim = ~isempty(labelDim);
end