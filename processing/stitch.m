clc; clear; close all;

% Extract two consecutive point clouds and use the first point cloud as
% reference.
ptCloudRef_struct = load(fullfile('sample_data', 'arrdata1.mat'));
ptCloudCurrent_struct = load(fullfile('sample_data', 'arrdata2.mat'));

ptCloudRef = pointCloud(ptCloudRef_struct.arr);
ptCloudCurrent = pointCloud(ptCloudCurrent_struct.arr);

gridSize = 0.5;
fixed = pcdownsample(ptCloudRef, 'gridAverage', gridSize);
moving = pcdownsample(ptCloudCurrent, 'gridAverage', gridSize);

% Note that the downsampling step does not only speed up the registration,
% but can also improve the accuracy.

tform = pcregistericp(moving, fixed, 'Metric','pointToPlane','Extrapolate', true);
ptCloudAligned = pctransform(ptCloudCurrent,tform);

mergeSize = 0.015;
ptCloudScene = pcmerge(ptCloudRef, ptCloudAligned, mergeSize);


% Visualize the world scene.
figure
subplot(2,2,1)
pcshow(ptCloudRef)
title('First pointcloud')
drawnow

subplot(2,2,3)
pcshow(ptCloudCurrent)
title('Second pointcloud')
drawnow

subplot(2,2,[2,4])
pcshow(ptCloudScene, 'VerticalAxis','Z', 'VerticalAxisDir', 'Up')
title('Merged pointclouds')
xlabel('X (mm)')
ylabel('Y (mm)')
zlabel('Z (mm)')
drawnow

