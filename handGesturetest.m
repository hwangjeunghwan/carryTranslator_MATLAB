function handGesturetest(net)
% function of handGesturetest()
%   A GUI based framework for hand Gesture recognition
%
    
    % Enable the kinect feature and initialize it
    
    colorVid = videoinput('kinect', 1);
    depthVid = videoinput('kinect', 2);
    triggerconfig(depthVid, 'manual');
    depthVid.FramesPerTrigger = 1;
    depthVid.TriggerRepeat = inf;
    
    triggerconfig(colorVid, 'manual');
    colorVid.FramesPerTrigger = 1;
    colorVid.TriggerRepeat = inf;
    set(getselectedsource(depthVid), 'TrackingMode', 'Skeleton');
    % timer handler for continuous operation
     t = timer('Period', 0.1,'ExecutionMode', 'fixedRate');
     t.TimerFcn = @dispDepth;
    window=figure('Color',[0.9255 0.9137 0.8471],'Name','Depth Camera',...
                  'DockControl','off','Units','Pixels',...
                  'toolbar','none',...
                  'Position',[50 50 800 600]);
             
    startb=uicontrol('Parent',window,'Style','pushbutton','String',...
                        'START',...
                        'FontSize',11 ,...
                        'Units','normalized',...
                        'Position',[0.22 0.02 0.16 0.08],...
                        'Callback',@startCallback);
    
    stopb=uicontrol('Parent',window,'Style','pushbutton','String',...
                        'STOP',...
                        'FontSize',11 ,...
                        'Units','normalized',...
                        'Position',[0.5 0.02 0.16 0.08],...
                        'Callback',@stopCallback);

    % main function for displaying depth
    function dispDepth(obj, event)
       % Generate color frame
       trigger(depthVid);
       trigger(colorVid);
       [depthMap, ~, depthMetaData] = getdata(depthVid);
       [colorFrameData] = getdata(colorVid);
       idx = find(depthMetaData.IsSkeletonTracked);
       subplot(2,2,1);
       imshow(colorFrameData);
       
       % generate depth frame and rescale it in 0~4096
       
       if idx ~= 0
           % Extract right hand position
           rightHand = depthMetaData.JointDepthIndices(12,:,idx);
           
           % Extract right hand realword position
           zCoord = 1e3*min(depthMetaData.JointWorldCoordinates(12,:,idx));
           radius = round(90 - zCoord / 50);
           rightHandBox = [rightHand-0.5*radius 1.2*radius 1.2*radius];
           % Define the region of interests (ROI) in right hand and segmented it
           rectangle('position', rightHandBox, 'EdgeColor', [1 1 0]);
           handDepthImage = imcrop(colorFrameData,rightHandBox);
           
           if ~isempty(handDepthImage)
               % preprocessing for background segmentation
                %imageSize = size(handDepthImage);
%                  for k = 1:imageSize(1)
%                      for j = 1:imageSize(2)
%                          if handDepthImage(k, j) > 2300
%                              handDepthImage(k, j) = 0;
%                          end
%                      end
%                  end
               % image resizing in 100*100 image
               temp = imresize(handDepthImage, [224 224]);
               
               % Extract HOG features using predefined function
               YPred = classify(net,temp);
               result = string(YPred);
               % show the result using svmstruct
               handDepthImage = insertText(handDepthImage,[0 0],result, 'Font','Dotum','FontSize', 55, 'BoxColor', 'black', 'TextColor','white');
               subplot(2,2,2);
               imshow(handDepthImage);

           end          
       end
    end
    
    % Callback handler for windows
    function startCallback(obj, event)
       start(depthVid);
       start(colorVid);
       start(t);
    end

    function stopCallback(obj, event)
       stop(depthVid);
       stop(colorVid);
       stop(t);
    end
end

