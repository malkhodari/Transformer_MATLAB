    function Z = iMergeHeads(X)
        % Merge the numFeatures-by-sequenceLength-by-numHeads array to a
        % (numFeatures*numHeads)-by-sequenceLength array
        X = permute(X,[1,3,2,4]);
        Z = reshape(X,size(X,1)*size(X,2),size(X,4),[]);
    end