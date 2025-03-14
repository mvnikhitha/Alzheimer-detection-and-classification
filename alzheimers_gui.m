function alzheimers_gui
    % Create GUI figure
    fig = figure('Name', 'Alzheimer''s Disease Classification', ...
                 'Position', [500, 300, 600, 400], ...
                 'Color', [0.7, 0.9, 1]);

    % Button background color (light yellow)
    buttonColor = [1, 1, 0.8];

    % UI Components
    uicontrol('Style', 'pushbutton', 'String', 'Load Image', ...
              'Position', [30, 300, 150, 40], 'BackgroundColor', buttonColor, ...
              'Callback', @loadImage);

    uicontrol('Style', 'pushbutton', 'String', 'Classify', ...
              'Position', [30, 240, 150, 40], 'BackgroundColor', buttonColor, ...
              'Callback', @classifyImage);
    
    % Axes for displaying the image on the right side
    ax = axes('Units', 'Pixels', 'Position', [220, 120, 350, 250]);
    imshow([]);

    % Text for displaying the classification result at the bottom middle
    resultText = uicontrol('Style', 'text', 'Position', [200, 70, 200, 30], ...
                           'String', 'Result: ', 'FontSize', 12, 'FontWeight', 'bold', ...
                           'BackgroundColor', [0.7, 0.9, 1]);

    % Exit button at the bottom
    uicontrol('Style', 'pushbutton', 'String', 'Exit', ...
              'Position', [250, 20, 100, 40], 'BackgroundColor', buttonColor, ...
              'Callback', @(~,~) close(fig));

    % Global variables
    global img net;
    img = [];

    % Load or train the model at startup
    if isfile('alzheimers_cnn.mat')
    load('alzheimers_cnn.mat', 'trainedNet');
    net = trainedNet; % Ensure correct assignment
        else
    net = trainCNNModel();
    save('alzheimers_cnn.mat', 'net');
    end

    % Function to load an image
    function loadImage(~, ~)
        [file, path] = uigetfile({'.jpg;.png;.jpeg;.tif', 'Image Files'});
        if isequal(file, 0)
            return;
        end
        img = imread(fullfile(path, file));
        img = preprocessImage(img);
        imshow(img, 'Parent', ax);
        set(resultText, 'String', 'Result: ');
    end

    % Function to classify an image
    function classifyImage(~, ~)
        if isempty(img)
            msgbox('Please load an image first.', 'Error', 'error');
            return;
        end

        % Predict using CNN
        predictedLabel = classify(net, img);
        set(resultText, 'String', ['Result: ', char(predictedLabel)]);
    end

    % Function to train the CNN model
    function trainedNet = trainCNNModel()
        datasetPath = 'D:\Downloads\archive (1)\Combined Dataset\train'; % Ensure this dataset exists
        if ~isfolder(datasetPath)
            error('Dataset folder not found.');
        end

        % Load dataset with error handling
        imds = imageDatastore(datasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames', ...
                              'ReadFcn', @safeReadImage);
        [trainImgs, testImgs] = splitEachLabel(imds, 0.8, 'randomized');

        % Load pre-trained MobileNetV2
        net = mobilenetv2;

        % Modify last layers for classification
        lgraph = layerGraph(net);

        % Remove last few layers (keep feature extractor)
        lgraph = removeLayers(lgraph, {'Logits', 'Logits_softmax', 'ClassificationLayer_Logits'});

        % Get the number of classes
        numClasses = numel(unique(imds.Labels));

        % Add new layers
        newLayers = [
            fullyConnectedLayer(numClasses, 'Name', 'new_fc')
            softmaxLayer('Name', 'new_softmax')
            classificationLayer('Name', 'new_class')];

        lgraph = addLayers(lgraph, newLayers);
        lgraph = connectLayers(lgraph, 'global_average_pooling2d_1', 'new_fc');

        % Training options
        options = trainingOptions('adam', ...
            'MaxEpochs', 5, ...  % Reduced for faster training
            'MiniBatchSize', 16, ... % Smaller batch size to reduce memory usage
            'ValidationData', testImgs, ...
            'ValidationFrequency', 5, ...
            'Verbose', true, ...
            'Plots', 'training-progress');

        % Train network
        trainedNet = trainNetwork(trainImgs, lgraph, options);
        save('alzheimers_cnn.mat', 'trainedNet');
    end

    % Function to safely read and preprocess an image
    function img = safeReadImage(filename)
        try
            img = imread(filename);
            img = preprocessImage(img);
        catch
            warning(['Skipping unreadable image: ', filename]);
            img = zeros(224, 224, 3, 'uint8'); % Return a blank image
        end
    end

    % Function to preprocess image for CNN input
    function img = preprocessImage(img)
        img = imresize(img, [224, 224]); % Resize
        if size(img, 3) == 1
            img = repmat(img, [1, 1, 3]); % Convert grayscale to RGB
        end
        img = single(img) / 255; % Normalize
    end
end