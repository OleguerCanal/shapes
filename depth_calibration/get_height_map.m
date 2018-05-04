clc; clear all;

dz_dx = load('dz_dx_mat.mat', 'arr');
dz_dx = dz_dx.arr;

mesh(dz_dx);
disp('Press a key To continue')
pause;


dz_dy = load('dz_dy_mat.mat', 'arr');
dz_dy = dz_dy.arr;

mesh(dz_dy);
disp('Press a key To continue')
pause;

img = fast_poisson2(dz_dx, dz_dy);
mesh(img);