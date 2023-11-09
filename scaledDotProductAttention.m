function A = scaledDotProductAttention(Q,K,V,numHeads,dropoutProb)

        % The function is vectorized to apply attention to all heads
        % simultaneously.

        channelDim = 1;
        observationDim = 2;
        positionDim = 3;
        [Q_edit, K_edit, V_edit] = iOrderForBuiltin(Q, K, V, channelDim, observationDim, positionDim);

        numFeatures = size(K_edit,1);

        % Matrix multiplication
        K2 = iSplitHeadsQ(K_edit,numFeatures,numHeads);
        Q2 = iSplitHeadsQ(Q_edit,numFeatures,numHeads);
        W2 = pagemtimes(K2, 'transpose', Q2, 'none');

        % Scale
        scale = 1/sqrt(size(K_edit,1)/numHeads);
        W2 = W2.*scale;
        W2 = W2 .* 1 - (1e10) .* (~1);

        % Softmax
        W2 = softmax(W2,DataFormat="CTUB");

        % DropOut
        if dropoutProb ~= 0
        [W2, ~] = iDropout(W2, dropoutProb);
        end
        
        % Matrix multiplication
        V2 = iSplitHeadsQ(V_edit,numFeatures,numHeads);
        A = pagemtimes(V2,W2);

end