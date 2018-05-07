clc; clear all;

dz_dx = load('dz_dx_mat.mat', 'arr');
dz_dx = dz_dx.arr;
 
% mesh(dz_dx);
% % imagesc(dz_dx)
% disp('Press a key To continue')
% pause;


dz_dy = load('dz_dy_mat.mat', 'arr');
dz_dy = dz_dy.arr;
 
% mesh(dz_dy);
% mesh(dz_dy)
% disp('Press a key To continue')
% pause;

img = fast_poisson2(dz_dy, dz_dx);
daspect([1 1 1])
mesh(img)
daspect([1 1 1])