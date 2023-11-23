% plot figures one by one
% Figure 5 (Fig5) - DN-PINNs Result
% To obtain the data represented in Figure 5, execute the script `DN-PINNs-2prob-2D.py`.
% Figure 6 (Fig6) - DNLA (PINNs) Results
% For the results displayed in Figure 6, execute the script `DNLM-2prob-2D-DNLM(PINN).py` to generate the corresponding data.
% Figure 7 (Fig7) - DNLA (Ritz) Results
% To generate the results shown in Figure 7, run the script `DNLM-2prob-2D-DNLM(Ritz).py`.
% Figure Generation:
% Utilize MATLAB and execute the script `plot_Solutions_1by1.m` to create the graphical representation associated with the data obtained from the previous steps.
% Please ensure that all necessary dependencies and prerequisites for the Python scripts and MATLAB files are correctly installed and configured before executing the scripts.

format short e
close all
path = 'Results\';
savepath = 'Results\';
if(exist(savepath,'dir')~=7)
    mkdir(savepath);
end
index = '13';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% problem setting
num_pts = 100;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% exact solution over entire domain
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
file = '\u_exact_sub1.mat';
load(strcat(path,file))
u_NN_left =  u_ex1;
file = '\u_exact_sub2.mat';
load(strcat(path,file))
u_NN_right = u_ex2;
% mesh
fig=figure('NumberTitle','off','Name','exact Solution','Renderer', 'painters', 'Position', [0 0 700 600]);

[V_mesh_left, qx_left, qy_left, dx_left, dy_left] = generate_mesh(0, 0.5, 0, 1, num_pts);
[V_mesh_right, qx_right, qy_right, dx_right, dy_right] = generate_mesh(0.5, 1, 0, 1, num_pts);
Ft = TriScatteredInterp(V_mesh_left(1,:)',V_mesh_left(2,:)',double(u_NN_left));
qz_left = Ft(qx_left,qy_left);
 
Ft = TriScatteredInterp(V_mesh_right(1,:)',V_mesh_right(2,:)',double(u_NN_right));
qz_right = Ft(qx_right,qy_right);

max_u_ite = double(max(max(u_NN_right),max(u_NN_left)));
min_u_ite = double(min(min(u_NN_right),min(u_NN_left)));

%---------------------%
axes4 = axes(fig);
axes4.Position(1,3) = axes4.Position(1,3)/7 * 5;
axes4.Position(1,2) = axes4.Position(1,4)/6*1;
axes4.Position(1,4) = axes4.Position(1,4)/6*5;
xticks([0  1]);
yticks([0  1]);
rectangle('position',[0 0 1 1] ,'LineWidth', 4);
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',25)
stringlabel = strcat(strcat('$\textnormal{Ite} = ',index),'$');
xlabel({stringlabel},'Interpreter','latex','FontSize',25)
axes2 = axes(fig);
axes2.Position(1,3) = axes2.Position(1,3)/14 * 5;
axes2.Position(1,2) = axes4.Position(1,2);
axes2.Position(1,4) = axes4.Position(1,4);

imagesc(axes2,dx_left,dy_left,qz_left);
colormap jet
caxis([min_u_ite max_u_ite])
colorbar off
axis off
axes3 = axes(fig);
axes3.Position(1,1) = axes2.Position(1,1) + axes2.Position(1,3);
axes3.Position(1,3) = axes3.Position(1,3)/14 * 5;
axes3.Position(1,2) = axes4.Position(1,2);
axes3.Position(1,4) = axes4.Position(1,4);
imagesc(axes3,dx_right,dy_right,qz_right)
colormap jet
caxis([min_u_ite max_u_ite])
axis off
h=colorbar('manual');
set(h,'Position',[0.72,0.13,0.025,0.7])
set(h,'Fontsize',25)
set(h,'LineWidth',2)
set(gcf,'color','w')
axis off
savefile = strcat(strcat('fig-DN-ex1-SOTA-DNLM-PINN-u-NN-ite-',index),'.png');
saveas(gcf,strcat(savepath,savefile));
%---------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Network predict solution final over entire domain
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
file = strcat(strcat('\u_NN_test_ite',index),'_sub1.mat');
load(strcat(path,file))
u_NN_left = u_NN_sub1;
%u_NN_left = u_ex1;
file = strcat(strcat('\u_NN_test_ite',index),'_sub2.mat');
load(strcat(path,file))
u_NN_right = u_NN_sub2;
% mesh
fig=figure('NumberTitle','off','Name','predict Solution','Renderer', 'painters', 'Position', [0 0 700 600]);

[V_mesh_left, qx_left, qy_left, dx_left, dy_left] = generate_mesh(0, 0.5, 0, 1, num_pts);
[V_mesh_right, qx_right, qy_right, dx_right, dy_right] = generate_mesh(0.5, 1, 0, 1, num_pts);
Ft = TriScatteredInterp(V_mesh_left(1,:)',V_mesh_left(2,:)',double(u_NN_left));
qz_left = Ft(qx_left,qy_left);
 
Ft = TriScatteredInterp(V_mesh_right(1,:)',V_mesh_right(2,:)',double(u_NN_right));
qz_right = Ft(qx_right,qy_right);

max_u_ite = double(max(max(u_NN_right),max(u_NN_left)));
min_u_ite = double(min(min(u_NN_right),min(u_NN_left)));

%---------------------%
axes4 = axes(fig);
axes4.Position(1,3) = axes4.Position(1,3)/7 * 5;
axes4.Position(1,2) = axes4.Position(1,4)/6*1;
axes4.Position(1,4) = axes4.Position(1,4)/6*5;
xticks([0  1]);
yticks([0  1]);
rectangle('position',[0 0 1 1] ,'LineWidth', 4);
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',25)
stringlabel = strcat(strcat('$\textnormal{Ite} = ',index),'$');
xlabel({stringlabel},'Interpreter','latex','FontSize',25)
axes2 = axes(fig);
axes2.Position(1,3) = axes2.Position(1,3)/14 * 5;
axes2.Position(1,2) = axes4.Position(1,2);
axes2.Position(1,4) = axes4.Position(1,4);

imagesc(axes2,dx_left,dy_left,qz_left);
colormap jet
caxis([min_u_ite max_u_ite])
colorbar off
axis off
axes3 = axes(fig);
axes3.Position(1,1) = axes2.Position(1,1) + axes2.Position(1,3);
axes3.Position(1,3) = axes3.Position(1,3)/14 * 5;
axes3.Position(1,2) = axes4.Position(1,2);
axes3.Position(1,4) = axes4.Position(1,4);
imagesc(axes3,dx_right,dy_right,qz_right)
colormap jet
caxis([min_u_ite max_u_ite])
axis off
h=colorbar('manual');
set(h,'Position',[0.72,0.13,0.025,0.7])
set(h,'Fontsize',25)
set(h,'LineWidth',2)
set(gcf,'color','w')
axis off
savefile = strcat(strcat('fig-DN-ex1-SOTA-DNLM-PINN-u-NN-ite-',index),'.png');
saveas(gcf,strcat(savepath,savefile));

%% pointwise error final over entire domain
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
file = strcat(strcat('\err_test_ite',index),'_sub1.mat');
load(strcat(path,file))
u_NN_left = abs(pointerr1);
%u_NN_left = zeros(size(pointerr1));
file = strcat(strcat('\err_test_ite',index),'_sub2.mat');
load(strcat(path,file))
u_NN_right = abs(pointerr2);
% mesh
fig=figure('NumberTitle','off','Name','pointwise error','Renderer', 'painters', 'Position', [0 0 700 600]);

[V_mesh_left, qx_left, qy_left, dx_left, dy_left] = generate_mesh(0, 0.5, 0, 1, num_pts);
[V_mesh_right, qx_right, qy_right, dx_right, dy_right] = generate_mesh(0.5, 1, 0, 1, num_pts);
Ft = TriScatteredInterp(V_mesh_left(1,:)',V_mesh_left(2,:)',double(u_NN_left));
qz_left = Ft(qx_left,qy_left);
 
Ft = TriScatteredInterp(V_mesh_right(1,:)',V_mesh_right(2,:)',double(u_NN_right));
qz_right = Ft(qx_right,qy_right);

max_u_ite = double(max(max(u_NN_right),max(u_NN_left)));
min_u_ite = double(min(min(u_NN_right),min(u_NN_left)));

%---------------------%
axes4 = axes(fig);
axes4.Position(1,3) = axes4.Position(1,3)/7 * 5;
axes4.Position(1,2) = axes4.Position(1,4)/6*1;
axes4.Position(1,4) = axes4.Position(1,4)/6*5;
xticks([0  1]);
yticks([0  1]);
rectangle('position',[0 0 1 1] ,'LineWidth', 4);
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',25)
stringlabel = strcat(strcat('$\textnormal{Ite} = ',index),'$');
xlabel({stringlabel},'Interpreter','latex','FontSize',25)
axes2 = axes(fig);
axes2.Position(1,3) = axes2.Position(1,3)/14 * 5;
axes2.Position(1,2) = axes4.Position(1,2);
axes2.Position(1,4) = axes4.Position(1,4);

imagesc(axes2,dx_left,dy_left,qz_left);
colormap jet
caxis([min_u_ite max_u_ite])
colorbar off
axis off
axes3 = axes(fig);
axes3.Position(1,1) = axes2.Position(1,1) + axes2.Position(1,3);
axes3.Position(1,3) = axes3.Position(1,3)/14 * 5;
axes3.Position(1,2) = axes4.Position(1,2);
axes3.Position(1,4) = axes4.Position(1,4);
imagesc(axes3,dx_right,dy_right,qz_right)
colormap jet
caxis([min_u_ite max_u_ite])
axis off
h=colorbar('manual');
set(h,'Position',[0.72,0.13,0.025,0.7])
set(h,'Fontsize',25)
set(h,'LineWidth',2)
set(gcf,'color','w')
axis off
savefile = strcat(strcat('fig-DN-ex1-SOTA-DNLM-PINN-pterr-ite-',index),'.png');
saveas(gcf,strcat(savepath,savefile));
%---------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Network predict gradient x on final over entire domain
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
file = strcat(strcat('\gradu_NN_test_ite',index),'_sub1.mat');
load(strcat(path,file))
u_NN_left = grad_u_test1(:,1);
file = strcat(strcat('\gradu_NN_test_ite',index),'_sub2.mat');
load(strcat(path,file))
u_NN_right = grad_u_test2(:,1);
% mesh
fig=figure('NumberTitle','off','Name','predict gradientx','Renderer', 'painters', 'Position', [0 0 700 600]);

[V_mesh_left, qx_left, qy_left, dx_left, dy_left] = generate_mesh(0, 0.5, 0, 1, num_pts);
[V_mesh_right, qx_right, qy_right, dx_right, dy_right] = generate_mesh(0.5, 1, 0, 1, num_pts);
Ft = TriScatteredInterp(V_mesh_left(1,:)',V_mesh_left(2,:)',double(u_NN_left));
qz_left = Ft(qx_left,qy_left);
 
Ft = TriScatteredInterp(V_mesh_right(1,:)',V_mesh_right(2,:)',double(u_NN_right));
qz_right = Ft(qx_right,qy_right);

max_u_ite = double(max(max(u_NN_right),max(u_NN_left)));
min_u_ite = double(min(min(u_NN_right),min(u_NN_left)));

%---------------------%
axes4 = axes(fig);
axes4.Position(1,3) = axes4.Position(1,3)/7 * 5;
axes4.Position(1,2) = axes4.Position(1,4)/6*1;
axes4.Position(1,4) = axes4.Position(1,4)/6*5;
xticks([0  1]);
yticks([0  1]);
rectangle('position',[0 0 1 1] ,'LineWidth', 4);
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',25)
stringlabel = strcat(strcat('$\textnormal{Ite} = ',index),'$');
xlabel({stringlabel},'Interpreter','latex','FontSize',25)
axes2 = axes(fig);
axes2.Position(1,3) = axes2.Position(1,3)/14 * 5;
axes2.Position(1,2) = axes4.Position(1,2);
axes2.Position(1,4) = axes4.Position(1,4);

imagesc(axes2,dx_left,dy_left,qz_left);
colormap jet
caxis([min_u_ite max_u_ite])
colorbar off
axis off
axes3 = axes(fig);
axes3.Position(1,1) = axes2.Position(1,1) + axes2.Position(1,3);
axes3.Position(1,3) = axes3.Position(1,3)/14 * 5;
axes3.Position(1,2) = axes4.Position(1,2);
axes3.Position(1,4) = axes4.Position(1,4);
imagesc(axes3,dx_right,dy_right,qz_right)
colormap jet
caxis([min_u_ite max_u_ite])
axis off
h=colorbar('manual');
set(h,'Position',[0.72,0.13,0.025,0.7])
set(h,'Fontsize',25)
set(h,'LineWidth',2)
set(gcf,'color','w')
axis off
savefile = strcat(strcat('fig-DN-ex1-SOTA-DNLM-PINN-grad_u_x-NN-ite-',index),'.png');
saveas(gcf,strcat(savepath,savefile));

%% pointwise error gradient x final over entire domain
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
file = strcat(strcat('\err_test_ite',index),'_sub1.mat');
load(strcat(path,'\gradu_exact_sub1.mat'))
pointerr1 = grad_u_test1(:,1)-gradu_Exact1(:,1);
u_NN_left = abs(pointerr1);
file = strcat(strcat('\err_test_ite',index),'_sub2.mat');
load(strcat(path,'\gradu_exact_sub2.mat'))
pointerr2 = grad_u_test2(:,1)-gradu_Exact2(:,1);
u_NN_right = abs(pointerr2);
% mesh
fig=figure('NumberTitle','off','Name','gradient x pointwise error','Renderer', 'painters', 'Position', [0 0 700 600]);

[V_mesh_left, qx_left, qy_left, dx_left, dy_left] = generate_mesh(0, 0.5, 0, 1, num_pts);
[V_mesh_right, qx_right, qy_right, dx_right, dy_right] = generate_mesh(0.5, 1, 0, 1, num_pts);
Ft = TriScatteredInterp(V_mesh_left(1,:)',V_mesh_left(2,:)',double(u_NN_left));
qz_left = Ft(qx_left,qy_left);
 
Ft = TriScatteredInterp(V_mesh_right(1,:)',V_mesh_right(2,:)',double(u_NN_right));
qz_right = Ft(qx_right,qy_right);

max_u_ite = double(max(max(u_NN_right),max(u_NN_left)));
min_u_ite = double(min(min(u_NN_right),min(u_NN_left)));

%---------------------%
axes4 = axes(fig);
axes4.Position(1,3) = axes4.Position(1,3)/7 * 5;
axes4.Position(1,2) = axes4.Position(1,4)/6*1;
axes4.Position(1,4) = axes4.Position(1,4)/6*5;
xticks([0  1]);
yticks([0  1]);
rectangle('position',[0 0 1 1] ,'LineWidth', 4);
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',25)
stringlabel = strcat(strcat('$\textnormal{Ite} = ',index),'$');
xlabel({stringlabel},'Interpreter','latex','FontSize',25)
axes2 = axes(fig);
axes2.Position(1,3) = axes2.Position(1,3)/14 * 5;
axes2.Position(1,2) = axes4.Position(1,2);
axes2.Position(1,4) = axes4.Position(1,4);

imagesc(axes2,dx_left,dy_left,qz_left);
colormap jet
caxis([min_u_ite max_u_ite])
colorbar off
axis off
axes3 = axes(fig);
axes3.Position(1,1) = axes2.Position(1,1) + axes2.Position(1,3);
axes3.Position(1,3) = axes3.Position(1,3)/14 * 5;
axes3.Position(1,2) = axes4.Position(1,2);
axes3.Position(1,4) = axes4.Position(1,4);
imagesc(axes3,dx_right,dy_right,qz_right)
colormap jet
caxis([min_u_ite max_u_ite])
axis off
h=colorbar('manual');
set(h,'Position',[0.72,0.13,0.025,0.7])
set(h,'Fontsize',25)
set(h,'LineWidth',2)
set(gcf,'color','w')
axis off
savefile = strcat(strcat('fig-DN-ex1-SOTA-DNLM-PINN-err-grad_u_x-ite-',index),'.png');
saveas(gcf,strcat(savepath,savefile));
%---------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [V_mesh, qx, qy, dx, dy] = generate_mesh(left, right, bottom, top, num_pts)

format short e

% original mesh
mesh_x = linspace(left, right, num_pts);
mesh_y = linspace(bottom, top, num_pts);
[mesh_x,mesh_y] = meshgrid(mesh_x,mesh_y);

V_mesh(1,:) = reshape(mesh_x,1,[]);
V_mesh(2,:) = reshape(mesh_y,1,[]);

% interpolate mesh
dx = left:0.002:right;
dy = bottom:0.002:top;
[qx,qy] = meshgrid(dx,dy);

end
