classdef posencLayer < nnet.layer.Layer
    % custom positional encoder layer.

    properties
        % Layer learnable parameters 
%         Weights 
%         Bias
%         Offset
%         ScaleFactor
posarray
    end
    
    methods
        function layer = posencLayer(numChannels,args) 
            % layer = posencLayer(args) creates a Positional encoder layer
            % with numChannels channels.
            %
            % layer = posencLayer(Name=name) also specifies the
            % layer name.
    
            arguments
%                 filterSize;
%                 numFilters;
                numChannels;
                args.Name = "";
            end
    
            % Set layer name.
            layer.Name = args.Name;

            % Set layer description.
            layer.Description = "Positional encoder layer";

            % Set layer type.
            layer.Type = "Positional Encoder";

            % Initializ weights, bias, offset, and scaleFactor.
%             layer.Weights = rand(filterSize,numChannels,numFilters);
%             layer.Bias = zeros(1,numChannels);
%             layer.Offset = zeros(1,numChannels);
%             layer.ScaleFactor = ones(1,numChannels);

            i = 1250;
            j = 1250; %%% can be more than the input so you take only what you need

            pos_array = zeros(i,1,j);

            for mini_id = 1:1

            for i_id = 1:i

                for j_id = 1:j

                    if mod(i_id,2) == 1
                       pos_array(j_id,mini_id,i_id) = cos(j_id / (10000^ ( (2*(i_id-1)) /1250 ) ) );
                    else
                       pos_array(j_id,mini_id,i_id) = sin(j_id / (10000^ ( (2*i_id) /1250 ) ) );
                    end


                end

            end

            end

            layer.posarray = pos_array;
%             A = pos_array(:,1,:);
%             A = reshape(A,[i j]);
%             figure;imagesc(A);

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

            Posaaray = layer.posarray;
            Posaaray = Posaaray(1:size(X,1),:,1:size(X,3));
            Posaaray2 = repmat(Posaaray,[1 size(X,2) 1]);

            Z = X + Posaaray2; %% INSTEAD just create a CxT array, then stack them to the size of minibatch

%             % 1-D convolution layer
%             weights = layer.Weights;
%             bias = layer.Bias;
%             Xe = dlconv(X,weights,bias,DataFormat="CBT",WeightsFormat="TUCU",Stride=1,Padding="same");
%             % GELU activation
%             Xe = 0.5*Xe.*(1 + tanh(sqrt(2/pi)*(Xe + 0.044715*(Xe.^3))));
%             % Residual connection and normalization
%             X = Xe + X;
%             offset = layer.Offset;
%             scaleFactor = layer.ScaleFactor;
%             Z = layernorm(X,offset,scaleFactor,DataFormat="CBT");
        end

        function Z = forward(layer, X)
%             % Forward input data through the layer at training
%             % time and output the result and a memory value.
%             %
%             % Inputs:
%             %         layer - Layer to forward propagate through 
%             %         X     - Input data
%             % Output:
%             %         Z - Output of layer forward function
% 
            Posaaray = layer.posarray;
            Posaaray = Posaaray(1:size(X,1),:,1:size(X,3));
            Posaaray2 = repmat(Posaaray,[1 size(X,2) 1]);

            Z = X + Posaaray2; %% INSTEAD just create a CxT array, then stack them to the size of minibatch

%             % 1-D convolution layer
%             weights = layer.Weights;
%             bias = layer.Bias;
%             Xe = dlconv(X,weights,bias,DataFormat="CBT",WeightsFormat="TUCU",Stride=1,Padding="same");
%             % GELU activation
%             Xe = 0.5*Xe.*(1 + tanh(sqrt(2/pi)*(Xe + 0.044715*(Xe.^3))));
%             % Residual connection and normalization
%             X = Xe + X;
%             offset = layer.Offset;
%             scaleFactor = layer.ScaleFactor;
%             Z = layernorm(X,offset,scaleFactor,DataFormat="CBT");
        end

    end
end