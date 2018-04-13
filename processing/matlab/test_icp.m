clc; clear; close all;

ptCloudRef_struct = load(fullfile('sample_data', 'arrdata1.mat'));
ptCloudCurrent_struct = load(fullfile('sample_data', 'arrdata2.mat'));

ptCloudRef = ptCloudRef_struct.arr;
ptCloudCurrent = ptCloudCurrent_struct.arr;

tol = 15
[R, t, corr, data2] = icp(ptCloudRef, ptCloudCurrent, tol)

% Visualize
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
pcshow(R, 'VerticalAxis','Z', 'VerticalAxisDir', 'Up')
title('Merged pointclouds')
xlabel('X (mm)')
ylabel('Y (mm)')
zlabel('Z (mm)')
drawnow