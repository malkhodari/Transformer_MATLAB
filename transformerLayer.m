classdef transformerLayer < nnet.layer.Layer 
    % custom transformer encoder layer.

    properties
        NumHeads;
    end

    properties (Learnable)
        % Layer learnable parameters 
        Weights_K
        Bias_K
        Weights_V
        Bias_V
        Weights_Q
        Bias_Q
        Weights_A
        Bias_A

        Weights_fc1
        Bias_fc1
        Weights_fc2
        Bias_fc2
%         Weights_fcz
%         Bias_fcz

        Offset_1
        ScaleFactor_1
        Offset_2
        ScaleFactor_2
    end
    
    methods
        function layer = transformerLayer(NumHeads,numChannels,args) 
            % layer = transformerLayer(args) creates a Transformer encoder layer
            % with numChannels channels.
            %
            % layer = transformerLayer(Name=name) also specifies the
            % layer name.
    
            arguments
                NumHeads;
                numChannels;
                args.Name = "";
            end
    
            % Set layer name.
            layer.Name = args.Name;

            % Set layer description.
            layer.Description = "Transformer encoder layer";
            
            % Set layer type.
            layer.Type = "Transformer Encoder";

            layer.NumHeads = NumHeads;

%             numIn = numChannels; 
%             scale = 0.01;
%             varWeights = 2 / ((1 + scale^2) * numIn); %leaky_HE
%             varWeights = 2 / (numIn); %HE %%% TRY USING GLOROT

numIn = numChannels;
numOut = numChannels;    
Z = 2*rand([numChannels numChannels]) - 1;
bound = sqrt(6 / (numIn + numOut));
% % 
weights = bound * Z;
weights = dlarray(weights);

%             weights = dlarray(randn([numChannels numChannels]) * sqrt(varWeights));
            layer.Weights_K = weights;
            layer.Bias_K = dlarray(zeros(1,numChannels));

%             weights = dlarray(randn([numChannels numChannels]) * sqrt(varWeights));
            layer.Weights_V = weights;
            layer.Bias_V = dlarray(zeros(1,numChannels));

%             weights = dlarray(randn([numChannels numChannels]) * sqrt(varWeights));
            layer.Weights_Q = weights;
            layer.Bias_Q = dlarray(zeros(1,numChannels));

%             weights = dlarray(randn([numChannels numChannels]) * sqrt(varWeights));
            layer.Weights_A = weights;
            layer.Bias_A = dlarray(zeros(1,numChannels));

%             weights = dlarray(randn([32 numChannels]) * sqrt(varWeights));
numIn = numChannels;
numOut = numChannels;    
Z = 2*rand([32 numChannels]) - 1;
bound = sqrt(6 / (numIn + numOut));
% 
weights = bound * Z;
weights = dlarray(weights);
            layer.Weights_fc1 = weights;
            layer.Bias_fc1 = dlarray(zeros(1,32));

%             weights = dlarray(randn([numChannels 32]) * sqrt(varWeights));
numIn = numChannels;
numOut = numChannels;    
Z = 2*rand([numChannels 32]) - 1;
bound = sqrt(6 / (numIn + numOut));
% 
weights = bound * Z;
weights = dlarray(weights);
            layer.Weights_fc2 = weights;
            layer.Bias_fc2 = dlarray(zeros(1,numChannels));

%             weights = dlarray(randn([numChannels numChannels]) * sqrt(varWeights));
% numIn = numChannels;
% numOut = numChannels;    
% Z = 2*rand([numChannels numChannels]) - 1;
% bound = sqrt(6 / (numIn + numOut));
% % 
% weights = bound * Z;
% weights = dlarray(weights);
%             layer.Weights_fcz = weights;
%             layer.Bias_fcz = dlarray(zeros(1,numChannels));

            layer.Offset_1 = dlarray(zeros(1,numChannels));
            layer.ScaleFactor_1 = dlarray(ones(1,numChannels));
            layer.Offset_2 = dlarray(zeros(1,numChannels));
            layer.ScaleFactor_2 = dlarray(ones(1,numChannels));
        end
        
        function Z = predict(layer, X)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer - Layer to forward propagate through 
            %         X     - Input data
            % Output:
            %         Z     - Output of layer forward function
            
            numHeads = layer.NumHeads;

            weights_K = layer.Weights_K;
            bias_K = layer.Bias_K;
            weights_V = layer.Weights_V;
            bias_V = layer.Bias_V;
            weights_Q = layer.Weights_Q;
            bias_Q = layer.Bias_Q;
            weights_A = layer.Weights_A;
            bias_A = layer.Bias_A;
            weights_fc1 = layer.Weights_fc1;
            bias_fc1 = layer.Bias_fc1;
            weights_fc2 = layer.Weights_fc2;
            bias_fc2 = layer.Bias_fc2;
%             weights_fcz = layer.Weights_fcz;
%             bias_fcz = layer.Bias_fcz;

            offset_1 = layer.Offset_1;
            scaleFactor_1 = layer.ScaleFactor_1;
            offset_2 = layer.Offset_2;
            scaleFactor_2 = layer.ScaleFactor_2;

            %ENCODERBLOCK Encoder block
                    % X = encoderBlock(X) passes Y through an encoder
                    % block.
            
                    % Multi-Head Attention               
                    % Linear
                    K = fullyconnect(X,weights_K,bias_K,DataFormat="CBT");
                    V = fullyconnect(X,weights_V,bias_V,DataFormat="CBT");
                    Q = fullyconnect(X,weights_Q,bias_Q,DataFormat="CBT");

                    % Scaled dot-product attention
                    dropoutProb = 0; %% no dropout in predict
                    A = scaledDotProductAttention(Q,K,V,numHeads,dropoutProb);

                    % Concatenation
                    A = iMergeHeads(A);
                    
%                     A = attention(Q,K,V,numHeads,DropoutProbability=0,DataFormat="CBT"); %%For MATLAB R2022b (Same result)

                    % Linear
                    xa = fullyconnect(A,weights_A,bias_A,DataFormat="CBT");

                    % Add & Norm
                    X = X + xa;
                    X = layernorm(X,offset_1,scaleFactor_1,DataFormat="CBT");
            
                    % Feed Forward
                    xff = fullyconnect(X,weights_fc1,bias_fc1,DataFormat="CBT");
                    % gelu activation
                    xff = 0.5*xff.*(1 + tanh(sqrt(2/pi)*(xff + 0.044715*(xff.^3))));
                    xff = fullyconnect(xff,weights_fc2,bias_fc2,DataFormat="CBT");
            
                    % Add & Norm
                    X = X + xff;
                    Z = layernorm(X,offset_2,scaleFactor_2,DataFormat="CBT");
            
%             Z = fullyconnect(X,weights_fcz,bias_fcz,DataFormat="CBT");

        end

        function Z = forward(layer, X)
            % Forward input data through the layer at training
            % time and output the result and a memory value.
            %
            % Inputs:
            %         layer - Layer to forward propagate through 
            %         X     - Input data
            % Output:
            %         Z - Output of layer forward function
            
            numHeads = layer.NumHeads;

            weights_K = layer.Weights_K;
            bias_K = layer.Bias_K;
            weights_V = layer.Weights_V;
            bias_V = layer.Bias_V;
            weights_Q = layer.Weights_Q;
            bias_Q = layer.Bias_Q;
            weights_A = layer.Weights_A;
            bias_A = layer.Bias_A;
            weights_fc1 = layer.Weights_fc1;
            bias_fc1 = layer.Bias_fc1;
            weights_fc2 = layer.Weights_fc2;
            bias_fc2 = layer.Bias_fc2;
%             weights_fcz = layer.Weights_fcz;
%             bias_fcz = layer.Bias_fcz;

            offset_1 = layer.Offset_1;
            scaleFactor_1 = layer.ScaleFactor_1;
            offset_2 = layer.Offset_2;
            scaleFactor_2 = layer.ScaleFactor_2;

            %ENCODERBLOCK Encoder block
                    % X = encoderBlock(X) passes Y through an encoder
                    % block.
            
                    % Multi-Head Attention               
                    % Linear
                    K = fullyconnect(X,weights_K,bias_K,DataFormat="CBT");
                    V = fullyconnect(X,weights_V,bias_V,DataFormat="CBT");
                    Q = fullyconnect(X,weights_Q,bias_Q,DataFormat="CBT");

                    % Scaled dot-product attention
                    dropoutProb = 0.2;
                    A = scaledDotProductAttention(Q,K,V,numHeads,dropoutProb);

                    % Concatenation
                    A = iMergeHeads(A);
                  
%                     A = attention(Q,K,V,numHeads,DropoutProbability=0.2,DataFormat="CBT"); %%For MATLAB R2022b (Same result)

                    % Linear
                    xa = fullyconnect(A,weights_A,bias_A,DataFormat="CBT");

                    % Add & Norm
                    X = X + xa;
                    X = layernorm(X,offset_1,scaleFactor_1,DataFormat="CBT"); %% Try custom normalization only on C
                
                    % Feed Forward
                    xff = fullyconnect(X,weights_fc1,bias_fc1,DataFormat="CBT");
                    % gelu activation
                    xff = 0.5*xff.*(1 + tanh(sqrt(2/pi)*(xff + 0.044715*(xff.^3))));
                    xff = fullyconnect(xff,weights_fc2,bias_fc2,DataFormat="CBT");
            
                    % Add & Norm
                    X = X + xff;
                    Z = layernorm(X,offset_2,scaleFactor_2,DataFormat="CBT");
            
%             Z = fullyconnect(X,weights_fcz,bias_fcz,DataFormat="CBT");

        end

    end
end