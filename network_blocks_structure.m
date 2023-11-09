function [lgraph] = network_blocks_structure(lgraph,firstnumfilt,label,block_id,initializer)

%%% CNN 1 layer
filterSize = 64;
numFilters = firstnumfilt;
stride = 2;
layer = [convolution1dLayer(filterSize,numFilters,Stride=stride,DilationFactor=1,Padding="same",WeightsInitializer=initializer,Name=['conv_1',label]),...
         batchNormalizationLayer(Name=['bn_1',label]),...
         geluLayer(Name=['gelu_1',label]),...
         maxPooling1dLayer(3,Stride=2,Name=['mxpool_1',label])];
lgraph = addLayers(lgraph,layer);
lgraph = connectLayers(lgraph,['input1'],['conv_1',label]);

%%% CNN 2 layer
filterSize = 32;
numFilters = firstnumfilt*2;
stride = 2;
layer = [convolution1dLayer(filterSize,numFilters,Stride=stride,DilationFactor=1,Padding="same",WeightsInitializer=initializer,Name=['conv_2',label]),...
         batchNormalizationLayer(Name=['bn_2',label]),...
         geluLayer(Name=['gelu_2',label]),...
         maxPooling1dLayer(3,Stride=2,Name=['mxpool_2',label])];
lgraph = addLayers(lgraph,layer);
lgraph = connectLayers(lgraph,['mxpool_1',label],['conv_2',label]);

%%% Positional encoder
numChannels = firstnumfilt*2;
layer_posenc = posencLayer(numChannels,Name=['posenc_1',label]);
lgraph = addLayers(lgraph,layer_posenc);
lgraph = connectLayers(lgraph,['mxpool_2',label],['posenc_1',label]);

%%% Transformer layer
NumBlocks = 1;
NumHeads = 2; 
for transblockid = 1:NumBlocks
layer_transformer = transformerLayer(NumHeads,numChannels,Name=['transformer_',num2str(transblockid),label]);
lgraph = addLayers(lgraph,layer_transformer);
if transblockid == 1
lgraph = connectLayers(lgraph,['posenc_1',label],['transformer_',num2str(transblockid),label]);
else
lgraph = connectLayers(lgraph,['transformer_',num2str(transblockid-1),label],['transformer_',num2str(transblockid),label]);
end
end

% %%% Self-attention pooling
% layer = [fullyConnectedLayer(firstnumfilt/2,Name=['fc_linear1',label]),...
%          softmaxLayer(Name=['sf_linear1',label])];
% lgraph = addLayers(lgraph,layer);
% lgraph = connectLayers(lgraph,['transformer_',num2str(transblockid),label],['fc_linear1',label]);
% mul_layer = multiplicationLayer(2,'Name',['mul_1',label]);
% lgraph = addLayers(lgraph,mul_layer);
% lgraph = connectLayers(lgraph,['transformer_',num2str(transblockid),label],['mul_1',label,'/in1']);
% lgraph = connectLayers(lgraph,['sf_linear1',label],['mul_1',label,'/in2']);

%%% Decoder
layer = [fullyConnectedLayer(firstnumfilt*2,WeightsInitializer=initializer,Name=['fc_linear2',label]),...
         dropoutLayer(0.05,'Name',['drop1',label])];
lgraph = addLayers(lgraph,layer);
lgraph = connectLayers(lgraph,['transformer_',num2str(transblockid),label],['fc_linear2',label]);

%%% Linear + Relu + dropout
layer = [fullyConnectedLayer(firstnumfilt*2,WeightsInitializer=initializer,Name=['fc_linear5',label]),...
         reluLayer(Name=['relu_5',label]),...
         dropoutLayer(0.05,'Name',['drop2',label]),...
         globalAveragePooling1dLayer("Name",['gapl',label])];
lgraph = addLayers(lgraph,layer);
lgraph = connectLayers(lgraph,['drop1',label],['fc_linear5',label]);

end

