function Z = iSplitHeads(X,numFeatures,numHeads)
            % For the key and value vectors, permute to put the dimension for
            % the heads last and the sequence length second. This enables
            % batched matrix multiplication to compute attention for all of the
            % heads at once.
            X2 = reshape(X,numFeatures/numHeads,numHeads,size(X,3),size(X,2));
            Z = permute(X2,[1,3,2,4]);
        end