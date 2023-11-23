% task4_show_results_ex2.m
% plot figures one by one with flower-interface

format short e
close all
% set the load path and save path
path = 'Results\';
savepath = 'Figures\';
% set the file name
savename = 'fig-DN-ex2f-';
algorithm = 'DNLM-DeepRitz-';
if(exist(savepath,'dir')~=7)
    mkdir(savepath);
end
index = '13';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% problem setting
num_pts = 100;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('flower-quarter-fig.mat')
T_flower_in = elem +1;
V_flower_in = node;
load('flower-quarter-out-fig.mat')
T_flower_out = elem + 1;
V_flower_out = node;
%% exact solution over entire domain
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% mesh

fig=figure('NumberTitle','off','Name','Exact Solution','Renderer', 'painters', 'Position', [0.01 0 700 600]);
axes4 = axes(fig);
axes4.Position(1,3) = axes4.Position(1,3)/7 * 5;
axes4.Position(1,2) = axes4.Position(1,4)/6*1;
axes4.Position(1,4) = axes4.Position(1,4)/6*5;
xlim([0.01 1.01]);
xticks([0.01 0.51 1.01]);
yticks([0  1]);
rectangle('position',[0.01 0 1 1] ,'LineWidth', 4);
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',25)
stringlabel = '$u(x,y)$';
xlabel({stringlabel},'Interpreter','latex','FontSize',22)

axes2=axes(fig);
axes2.Position(1,3) = axes2.Position(1,3)/7*5;
axes2.Position(1,2) = axes4.Position(1,2);
axes2.Position(1,4) = axes4.Position(1,4);
axes2.Position(1,1) = axes4.Position(1,1) - 0.0025;
file = '\u_exact_sub1.mat';
load(strcat(path,file))
file = '\u_exact_sub2.mat';
load(strcat(path,file))


predict_out = u_ex2';
predict_in =  u_ex1';
patch('Faces',T_flower_in','Vertices',V_flower_in','FaceVertexCData', predict_in, 'FaceColor', 'interp', 'LineStyle','none');
patch('Faces',T_flower_out','Vertices',V_flower_out','FaceVertexCData', predict_out, 'FaceColor', 'interp', 'LineStyle','none');
%---------------------%

axis off
colormap jet
h=colorbar('manual');
set(h,'Position',[0.72,0.13,0.02,0.69])
set(h,'Fontsize',25)
set(gcf,'color','w')
axis off
set(gca,'FontSize',18);
savefile = strcat(strcat(savename,'u-exact'),'.png');
saveas(gcf,strcat(savepath,savefile));
%---------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% exact gradient_x solution over entire domain
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% mesh

fig=figure('NumberTitle','off','Name','Exact Solution dx','Renderer', 'painters', 'Position', [0.01 0 700 600]);
axes4 = axes(fig);
axes4.Position(1,3) = axes4.Position(1,3)/7 * 5;
axes4.Position(1,2) = axes4.Position(1,4)/6*1;
axes4.Position(1,4) = axes4.Position(1,4)/6*5;
xlim([0.01 1.01]);
xticks([0.01 0.51 1.01]);
yticks([0  1]);
rectangle('position',[0.01 0 1 1] ,'LineWidth', 4);
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',25)
stringlabel = '$\partial_x u(x,y)$';
xlabel({stringlabel},'Interpreter','latex','FontSize',22)

axes2=axes(fig);
axes2.Position(1,3) = axes2.Position(1,3)/7*5;
axes2.Position(1,2) = axes4.Position(1,2);
axes2.Position(1,4) = axes4.Position(1,4);
axes2.Position(1,1) = axes4.Position(1,1) - 0.0025;
file = '\gradu_exact_sub2.mat';
load(strcat(path,file))
file = '\gradu_exact_sub1.mat';
load(strcat(path,file))


predict_out = gradu_Exact2(:,1);
predict_in =  gradu_Exact1(:,1);
patch('Faces',T_flower_in','Vertices',V_flower_in','FaceVertexCData', predict_in, 'FaceColor', 'interp', 'LineStyle','none');
patch('Faces',T_flower_out','Vertices',V_flower_out','FaceVertexCData', predict_out, 'FaceColor', 'interp', 'LineStyle','none');
%---------------------%

axis off
colormap jet
h=colorbar('manual');
set(h,'Position',[0.72,0.13,0.02,0.69])
set(h,'Fontsize',25)
set(gcf,'color','w')
axis off
set(gca,'FontSize',18);
savefile = strcat(strcat(savename,'exact-u-dx'),'.png');
saveas(gcf,strcat(savepath,savefile));
%---------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig=figure('NumberTitle','off','Name','Exact Solution dy','Renderer', 'painters', 'Position', [0.01 0 700 600]);
axes4 = axes(fig);
axes4.Position(1,3) = axes4.Position(1,3)/7 * 5;
axes4.Position(1,2) = axes4.Position(1,4)/6*1;
axes4.Position(1,4) = axes4.Position(1,4)/6*5;
xlim([0.01 1.01]);
xticks([0.01 0.51 1.01]);
yticks([0  1]);
rectangle('position',[0.01 0 1 1] ,'LineWidth', 4);
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',25)
stringlabel = '$\partial_y u(x,y)$';
xlabel({stringlabel},'Interpreter','latex','FontSize',22)

axes2=axes(fig);
axes2.Position(1,3) = axes2.Position(1,3)/7*5;
axes2.Position(1,2) = axes4.Position(1,2);
axes2.Position(1,4) = axes4.Position(1,4);
axes2.Position(1,1) = axes4.Position(1,1) - 0.0025;
file = '\gradu_exact_sub2.mat';
load(strcat(path,file))
file = '\gradu_exact_sub1.mat';
load(strcat(path,file))


predict_out = gradu_Exact2(:,2);
predict_in =  gradu_Exact1(:,2);
patch('Faces',T_flower_in','Vertices',V_flower_in','FaceVertexCData', predict_in, 'FaceColor', 'interp', 'LineStyle','none');
patch('Faces',T_flower_out','Vertices',V_flower_out','FaceVertexCData', predict_out, 'FaceColor', 'interp', 'LineStyle','none');
%---------------------%

axis off
colormap jet
h=colorbar('manual');
set(h,'Position',[0.72,0.13,0.02,0.69])
set(h,'Fontsize',25)
set(gcf,'color','w')
axis off
set(gca,'FontSize',18);
savefile = strcat(strcat(savename,'exact-u-dy'),'.png');
saveas(gcf,strcat(savepath,savefile));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Network predict solution the first iteration over entire domain
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig=figure('NumberTitle','off','Name','Exact Solution','Renderer', 'painters', 'Position', [0.01 0 700 600]);
axes4 = axes(fig);
axes4.Position(1,3) = axes4.Position(1,3)/7 * 5;
axes4.Position(1,2) = axes4.Position(1,4)/6*1;
axes4.Position(1,4) = axes4.Position(1,4)/6*5;
xticks([0.01  0.51  1.01]);
yticks([0  1]);
rectangle('position',[0.01 0 1 1] ,'LineWidth', 4);
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',25)
stringlabel = strcat(strcat('$\textnormal{Ite} = ',index),'$');
xlabel({stringlabel},'Interpreter','latex','FontSize',25)

axes2=axes(fig);
axes2.Position(1,3) = axes2.Position(1,3)/7*5;
axes2.Position(1,2) = axes4.Position(1,2);
axes2.Position(1,4) = axes4.Position(1,4); 
axes2.Position(1,1) = axes4.Position(1,1) - 0.0025;
file = strcat(strcat('\u_NN_test_ite',index),'_sub2.mat');
load(strcat(path,file))
file = strcat(strcat('\u_NN_test_ite',index),'_sub1.mat');
load(strcat(path,file))


predict_out = u_NN_sub2';
predict_in =  u_NN_sub1';
patch('Faces',T_flower_in','Vertices',V_flower_in','FaceVertexCData', predict_in', 'FaceColor', 'interp', 'LineStyle','none');
patch('Faces',T_flower_out','Vertices',V_flower_out','FaceVertexCData', predict_out', 'FaceColor', 'interp', 'LineStyle','none');
%---------------------%

axis off
colormap jet
h=colorbar('manual');
set(h,'Position',[0.72,0.13,0.02,0.69])
set(h,'Fontsize',25)
set(gcf,'color','w')
axis off
set(gca,'FontSize',18);
savefile = strcat(strcat(strcat(savename,strcat(algorithm,strcat('u-NN-','ite-'))),index),'.png');
saveas(gcf,strcat(savepath,savefile));
%---------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% pointwise error first iteration over entire domain
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig=figure('NumberTitle','off','Name','pterr error','Renderer', 'painters', 'Position', [0 0 700 600]);
axes4 = axes(fig);
axes4.Position(1,3) = axes4.Position(1,3)/7 * 5;
axes4.Position(1,2) = axes4.Position(1,4)/6*1;
axes4.Position(1,4) = axes4.Position(1,4)/6*5;
xticks([0.01  0.51  1.01]);
yticks([0  1]);
rectangle('position',[0.01 0 1 1] ,'LineWidth', 4);
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',25)

% stringlabel = strcat(strcat('$\textnormal{Ite} = ',index),'$');
% xlabel({stringlabel},'Interpreter','latex','FontSize',25)

axes2=axes(fig);
axes2.Position(1,3) = axes2.Position(1,3)/7*5;
axes2.Position(1,2) = axes4.Position(1,2);
axes2.Position(1,4) = axes4.Position(1,4); 
axes2.Position(1,1) = axes4.Position(1,1) - 0.0025;
file = strcat(strcat('\err_test_ite',index),'_sub2.mat');
load(strcat(path,file))
file = strcat(strcat('\err_test_ite',index),'_sub1.mat');
load(strcat(path,file))

err_out = abs(pointerr2)';
err_in =  abs(pointerr1)';

% err_in = zeros('like', err_in);
patch('Faces',T_flower_in','Vertices',V_flower_in','FaceVertexCData', err_in, 'FaceColor', 'interp', 'LineStyle','none');
patch('Faces',T_flower_out','Vertices',V_flower_out','FaceVertexCData', err_out, 'FaceColor', 'interp', 'LineStyle','none');
%---------------------%

axis off
colormap jet
h=colorbar('manual');
set(h,'Position',[0.72,0.13,0.02,0.69])
set(h,'Fontsize',25)
set(gcf,'color','w')
axis off
set(gca,'FontSize',18);
savefile = strcat(strcat(strcat(savename,strcat(algorithm,strcat('pterr-','ite-'))),index),'.png');
saveas(gcf,strcat(savepath,savefile));
%---------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% %% Network predict gradient x on final over entire domain
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fig=figure('NumberTitle','off','Name','predict gradient_x','Renderer', 'painters', 'Position', [0 0 700 600]);
% axes4 = axes(fig);
% axes4.Position(1,3) = axes4.Position(1,3)/7 * 5;
% axes4.Position(1,2) = axes4.Position(1,4)/6*1;
% axes4.Position(1,4) = axes4.Position(1,4)/6*5;
% xticks([0.01  0.51  1.01]);
% yticks([0  1]);
% rectangle('position',[0.01 0 1 1] ,'LineWidth', 4);
% a = get(gca,'XTickLabel');
% set(gca,'XTickLabel',a,'FontName','Times','fontsize',25)
% % stringlabel = strcat(strcat('$\textnormal{Ite} = ',index),'$');
% % xlabel({stringlabel},'Interpreter','latex','FontSize',25)
% 
% axes2=axes(fig);
% axes2.Position(1,3) = axes2.Position(1,3)/7*5;
% axes2.Position(1,2) = axes4.Position(1,2);
% axes2.Position(1,4) = axes4.Position(1,4); axes2.Position(1,1) = axes4.Position(1,1) - 0.0025;
% file = strcat(strcat('\gradu_NN_ite',index),'_test_left.mat');
% load(strcat(path,file))
% file = strcat(strcat('\gradu_NN_ite',index),'_test_right.mat');
% load(strcat(path,file))
% 
% grad_u_x_left = grad_u_test1(:,1)';
% grad_u_x_right = grad_u_test2(:,1)';
% patch('Faces',T_flower_in','Vertices',V_flower_in','FaceVertexCData', grad_u_x_right', 'FaceColor', 'interp', 'LineStyle','none');
% patch('Faces',T_flower_out','Vertices',V_flower_out','FaceVertexCData', grad_u_x_left', 'FaceColor', 'interp', 'LineStyle','none');
% %---------------------%
% 
% axis off
% colormap jet
% h=colorbar('manual');
% set(h,'Position',[0.72,0.13,0.02,0.69])
% set(h,'Fontsize',25)
% set(gcf,'color','w')
% axis off
% set(gca,'FontSize',18);
% savefile = strcat(strcat(strcat(savename,'-grad_u_x-ite-'),index),'.png');
% saveas(gcf,strcat(savepath,savefile));
% 
% %% pointwise error gradient x final over entire domain
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% fig=figure('NumberTitle','off','Name','pterr gradient_x','Renderer', 'painters', 'Position', [0 0 700 600]);
% axes4 = axes(fig);
% axes4.Position(1,3) = axes4.Position(1,3)/7 * 5;
% axes4.Position(1,2) = axes4.Position(1,4)/6*1;
% axes4.Position(1,4) = axes4.Position(1,4)/6*5;
% xticks([0.01  0.51  1.01]);
% yticks([0  1]);
% rectangle('position',[0.01 0 1 1] ,'LineWidth', 4);
% a = get(gca,'XTickLabel');
% set(gca,'XTickLabel',a,'FontName','Times','fontsize',25)
% % stringlabel = strcat(strcat('$\textnormal{Ite} = ',index),'$');
% % xlabel({stringlabel},'Interpreter','latex','FontSize',25)
% 
% axes2=axes(fig);
% axes2.Position(1,3) = axes2.Position(1,3)/7*5;
% axes2.Position(1,2) = axes4.Position(1,2);
% axes2.Position(1,4) = axes4.Position(1,4); axes2.Position(1,1) = axes4.Position(1,1) - 0.0025;
% file = '\gradu_exact_right.mat';
% load(strcat(path,file))
% file = '\gradu_exact_left.mat';
% load(strcat(path,file))
% 
% err_u_x_left = abs(grad_u_test1(:,1)'-gradu_Exact1(:,1)');
% err_u_x_right = abs(grad_u_test2(:,1)'-gradu_Exact2(:,1)');
% patch('Faces',T_flower_in','Vertices',V_flower_in','FaceVertexCData', err_u_x_right', 'FaceColor', 'interp', 'LineStyle','none');
% patch('Faces',T_flower_out','Vertices',V_flower_out','FaceVertexCData', err_u_x_left', 'FaceColor', 'interp', 'LineStyle','none');
% %---------------------%
% 
% axis off
% colormap jet
% h=colorbar('manual');
% set(h,'Position',[0.72,0.13,0.02,0.69])
% set(h,'Fontsize',25)
% set(gcf,'color','w')
% axis off
% set(gca,'FontSize',18);
% savefile = strcat(strcat(strcat(savename,'-pterr-grad_u_x-ite-'),index),'.png');
% saveas(gcf,strcat(savepath,savefile));
% %---------------------%
% 
% 
% %% Network predict gradient y on final over entire domain
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fig=figure('NumberTitle','off','Name','predict gradient_y','Renderer', 'painters', 'Position', [0 0 700 600]);
% axes4 = axes(fig);
% axes4.Position(1,3) = axes4.Position(1,3)/7 * 5;
% axes4.Position(1,2) = axes4.Position(1,4)/6*1;
% axes4.Position(1,4) = axes4.Position(1,4)/6*5;
% xticks([0.01  0.51  1.01]);
% yticks([0  1]);
% rectangle('position',[0.01 0 1 1] ,'LineWidth', 4);
% a = get(gca,'XTickLabel');
% set(gca,'XTickLabel',a,'FontName','Times','fontsize',25)
% % stringlabel = strcat(strcat('$\textnormal{Ite} = ',index),'$');
% % xlabel({stringlabel},'Interpreter','latex','FontSize',25)
% 
% axes2=axes(fig);
% axes2.Position(1,3) = axes2.Position(1,3)/7*5;
% axes2.Position(1,2) = axes4.Position(1,2);
% axes2.Position(1,4) = axes4.Position(1,4); axes2.Position(1,1) = axes4.Position(1,1) - 0.0025;
% file = strcat(strcat('\gradu_NN_ite',index),'_test_left.mat');
% load(strcat(path,file))
% file = strcat(strcat('\gradu_NN_ite',index),'_test_right.mat');
% load(strcat(path,file))
% 
% grad_u_y_left = grad_u_test1(:,2)';
% grad_u_y_right = grad_u_test2(:,2)';
% patch('Faces',T_flower_in','Vertices',V_flower_in','FaceVertexCData', grad_u_y_right', 'FaceColor', 'interp', 'LineStyle','none');
% patch('Faces',T_flower_out','Vertices',V_flower_out','FaceVertexCData', grad_u_y_left', 'FaceColor', 'interp', 'LineStyle','none');
% %---------------------%
% 
% axis off
% colormap jet
% h=colorbar('manual');
% set(h,'Position',[0.72,0.13,0.02,0.69])
% set(h,'Fontsize',25)
% set(gcf,'color','w')
% axis off
% set(gca,'FontSize',18);
% savefile = strcat(strcat(strcat(savename,'-grad_u_y-ite-'),index),'.png');
% saveas(gcf,strcat(savepath,savefile));
% 
% %% pointwise error gradient y final over entire domain
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% fig=figure('NumberTitle','off','Name','pterr gradient_y','Renderer', 'painters', 'Position', [0 0 700 600]);
% axes4 = axes(fig);
% axes4.Position(1,3) = axes4.Position(1,3)/7 * 5;
% axes4.Position(1,2) = axes4.Position(1,4)/6*1;
% axes4.Position(1,4) = axes4.Position(1,4)/6*5;
% xticks([0.01  0.51  1.01]);
% yticks([0  1]);
% rectangle('position',[0.01 0 1 1] ,'LineWidth', 4);
% a = get(gca,'XTickLabel');
% set(gca,'XTickLabel',a,'FontName','Times','fontsize',25)
% % stringlabel = strcat(strcat('$\textnormal{Ite} = ',index),'$');
% % xlabel({stringlabel},'Interpreter','latex','FontSize',25)
% 
% axes2=axes(fig);
% axes2.Position(1,3) = axes2.Position(1,3)/7*5;
% axes2.Position(1,2) = axes4.Position(1,2);
% axes2.Position(1,4) = axes4.Position(1,4); axes2.Position(1,1) = axes4.Position(1,1) - 0.0025;
% file = '\gradu_exact_right.mat';
% load(strcat(path,file))
% file = '\gradu_exact_left.mat';
% load(strcat(path,file))
% 
% err_u_y_left = abs(grad_u_test1(:,2)'-gradu_Exact1(:,2)');
% err_u_y_right = abs(grad_u_test2(:,2)'-gradu_Exact2(:,2)');
% patch('Faces',T_flower_in','Vertices',V_flower_in','FaceVertexCData', err_u_y_right', 'FaceColor', 'interp', 'LineStyle','none');
% patch('Faces',T_flower_out','Vertices',V_flower_out','FaceVertexCData', err_u_y_left', 'FaceColor', 'interp', 'LineStyle','none');
% %---------------------%
% 
% axis off
% colormap jet
% h=colorbar('manual');
% set(h,'Position',[0.72,0.13,0.02,0.69])
% set(h,'Fontsize',25)
% set(gcf,'color','w')
% axis off
% set(gca,'FontSize',18);
% savefile = strcat(strcat(strcat(savename,'-pterr-grad_u_y-ite-'),index),'.png');
% saveas(gcf,strcat(savepath,savefile));
% %---------------------%
%% pointwise error H1 final over entire domain
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
file = strcat(strcat('\gradu_NN_test_ite',index),'_sub1.mat');
load(strcat(path,file))
file = strcat(strcat('\gradu_NN_test_ite',index),'_sub2.mat');
load(strcat(path,file))
file = '\gradu_exact_sub2.mat';
load(strcat(path,file))
file = '\gradu_exact_sub1.mat';
load(strcat(path,file))
fig=figure('NumberTitle','off','Name','pterr H1','Renderer', 'painters', 'Position', [0 0 700 600]);
axes4 = axes(fig);
axes4.Position(1,3) = axes4.Position(1,3)/7 * 5;
axes4.Position(1,2) = axes4.Position(1,4)/6*1;
axes4.Position(1,4) = axes4.Position(1,4)/6*5;
xticks([0.01  0.51  1.01]);
yticks([0  1]);
rectangle('position',[0.01 0 1 1] ,'LineWidth', 4);
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',25)
% stringlabel = strcat(strcat('$\textnormal{Ite} = ',index),'$');
% xlabel({stringlabel},'Interpreter','latex','FontSize',25)

axes2=axes(fig);
axes2.Position(1,3) = axes2.Position(1,3)/7*5;
axes2.Position(1,2) = axes4.Position(1,2);
axes2.Position(1,4) = axes4.Position(1,4); axes2.Position(1,1) = axes4.Position(1,1) - 0.0025;
file = '\gradu_exact_sub2.mat';
load(strcat(path,file))
file = '\gradu_exact_sub1.mat';
load(strcat(path,file))

err_H1_in = ((grad_u_test1(:,2)'-gradu_Exact1(:,2)').^2+(grad_u_test1(:,1)'-gradu_Exact1(:,1)').^2).^0.5;
err_H1_out = ((grad_u_test2(:,2)'-gradu_Exact2(:,2)').^2+(grad_u_test2(:,1)'-gradu_Exact2(:,1)').^2).^0.5;
patch('Faces',T_flower_in','Vertices',V_flower_in','FaceVertexCData', err_H1_in', 'FaceColor', 'interp', 'LineStyle','none');
patch('Faces',T_flower_out','Vertices',V_flower_out','FaceVertexCData', err_H1_out', 'FaceColor', 'interp', 'LineStyle','none');
%---------------------%

axis off
colormap jet
h=colorbar('manual');
set(h,'Position',[0.72,0.13,0.02,0.69])
set(h,'Fontsize',25)
set(gcf,'color','w')
axis off
set(gca,'FontSize',18);
savefile = strcat(strcat(strcat(savename,strcat(algorithm,strcat('pterr-H1-','ite-'))),index),'.png');
saveas(gcf,strcat(savepath,savefile));
%---------------------%
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
