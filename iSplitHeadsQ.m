function Z = iSplitHeadsQ(X,numFeatures,numHeads)
         X2 = reshape(X,numFeatures/numHeads,numHeads,size(X,2),[]);
         Z = permute(X2,[1,3,2,4]);
end