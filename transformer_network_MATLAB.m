function [lgraph] = transformer_network_MATLAB(NumSignals,Y_train,weights)

%%% Input 1 layer
layer = sequenceInputLayer(NumSignals,MinLength=20000,Normalization="zscore",Name="input1");
lgraph = layerGraph(layer);

initializer = 'glorot';

%%%%%%% Select only used features
block_id = 1;
label = ['_block',num2str(block_id)];
firstnumfilt = 32;
[lgraph] = network_blocks_structure(lgraph,firstnumfilt,label,block_id,initializer);


%%% Final layer
layer = [fullyConnectedLayer(length(unique(Y_train)),WeightsInitializer=initializer,Name="fc_linear6"),...
         softmaxLayer(Name="sf_final"),...
         classificationLayer('Classes',unique(Y_train),'ClassWeights',weights,'Name','classification')];

lgraph = addLayers(lgraph,layer);
lgraph = connectLayers(lgraph,['gapl',label],"fc_linear6");
 
analyzeNetwork(lgraph)


















% if NumSignals > 4
% %% Feature input layer (Comment if not needed)
% %%%%%%% Channel separation 
% selected_channel = 5;
% layer = channelSeparationLayer(selected_channel,['channel_',num2str(selected_channel),'_splitter']);
% lgraph = addLayers(lgraph,layer);
% lgraph = connectLayers(lgraph,'input1',['channel_',num2str(selected_channel),'_splitter']);
% 
% %%%%%%% Select only used features
% numextractedFeatures = 6;
% layer = featureSelectionLayer(numextractedFeatures,'feature_selection'); 
% lgraph = addLayers(lgraph,layer);
% lgraph = connectLayers(lgraph,['channel_',num2str(selected_channel),'_splitter'],'feature_selection');

% %%%%%%% Reshape to channel - batch
% layer = ReshapeLayer("Name","reshape");
% lgraph = addLayers(lgraph,layer);
% lgraph = connectLayers(lgraph,'feature_selection','reshape');
% 
% %%%%%%% Concatenate with transformer last layer
% dimension = 1;
% NumInputs = 2;
% input_stream_2 = dimConcatLayer(NumInputs,dimension,'input_stream_2');
% lgraph = addLayers(lgraph,input_stream_2);
% lgraph = connectLayers(lgraph,'input_stream_1',['input_stream_2/in',num2str(1)]);
% lgraph = connectLayers(lgraph,'reshape',['input_stream_2/in',num2str(2)]);
% lgraph = disconnectLayers(lgraph,'input_stream_1','fc_linear6');
% 
% %%%%%%% Normalization
% % layer = batchNormalizationLayer('Name','layernorm');
% % lgraph = addLayers(lgraph,layer);
% lgraph = connectLayers(lgraph,'input_stream_2','fc_linear6');
% % lgraph = connectLayers(lgraph,'layernorm','fc_linear6');
% 
% % analyzeNetwork(lgraph)
% 
% end

end

