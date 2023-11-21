% plot figures one by one on 4 sub domain

format short e
close all
path = 'D:\博士研究生\研二\Prj22-DDLM-V1-Code\main-DNLM-PINN\Codes\Results\2_4Prob-2D-4NN\DNLM(PINN)\231101\simulation-test';
savepath = 'D:\博士研究生\highcontrast\Figures\2_4Prob-2D\DN-PINNs\simulation-1\';
savename = 'fig-DN-ex3-';
algorithm = 'DN-PINNs-';

if(exist(savepath,'dir')~=7)
    mkdir(savepath);
end
index = 1;
index_1 = index+1;
index = num2str(index);
index_1 = num2str(index_1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% problem setting
num_pts = 100;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% exact solution over entire domain
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig=figure('NumberTitle','off','Name','Exact Solution','Renderer', 'painters', 'Position', [0 0 700 500]);


load(strcat(path,'\u_exact_sub1.mat'))
u_NN_1 = u_ex1;
load(strcat(path,'\u_exact_sub2.mat'))
u_NN_2 = u_ex2;
load(strcat(path,'\u_exact_sub3.mat'))
u_NN_3 = u_ex3;
load(strcat(path,'\u_exact_sub4.mat'))
u_NN_4 = u_ex4;

u_NN_1 = u_NN_1(10000:-1:1);
u_NN_2 = u_NN_2(10000:-1:1);
u_NN_3 = u_NN_3(10000:-1:1);
u_NN_4 = u_NN_4(10000:-1:1);
% mesh
[V_mesh_1, qx_1, qy_1, dx_1, dy_1] = generate_mesh(0, 0.5, 0.5, 1, num_pts);
[V_mesh_3, qx_3, qy_3, dx_3, dy_3] = generate_mesh(0.5, 1, 0, 0.5, num_pts);

[V_mesh_2, qx_2, qy_2, dx_2, dy_2] = generate_mesh(0, 0.5, 0, 0.5, num_pts);
[V_mesh_4, qx_4, qy_4, dx_4, dy_4] = generate_mesh(0.5, 1, 0.5, 1, num_pts);

dx_1 = dx_1(end:-1:1);
dx_3 = dx_3(end:-1:1);
dx_2 = dx_2(end:-1:1);
dx_4 = dx_4(end:-1:1);


Ft_1 = TriScatteredInterp(V_mesh_1(1,:)',V_mesh_1(2,:)',double(u_NN_1));
qz_1 = Ft_1(qx_1,qy_1);

Ft_2 = TriScatteredInterp(V_mesh_3(1,:)',V_mesh_3(2,:)',double(u_NN_3));
qz_3 = Ft_2(qx_3,qy_3);

Ft_3 = TriScatteredInterp(V_mesh_2(1,:)',V_mesh_2(2,:)',double(u_NN_2));
qz_2 = Ft_3(qx_2,qy_2);

Ft_4 = TriScatteredInterp(V_mesh_4(1,:)',V_mesh_4(2,:)',double(u_NN_4));
qz_4 = Ft_4(qx_4,qy_4);

% Ft = TriScatteredInterp(V_mesh_B(1,:)',V_mesh_B(2,:)',double(u_NN_2));
% qz_B = Ft(qx_B,qy_B);

max_u_ite = double(max(max(u_NN_1),max(u_NN_2)));
min_u_ite = double(min(min(u_NN_1),min(u_NN_2)));

%---------------------%
axes1 = axes(fig);
axes1.Position(1,3)=axes1.Position(1,3)/14*5;
axes1.Position(1,4)=axes1.Position(1,4)/2;
axes1.Position(1,2)=axes1.Position(1,4)+axes1.Position(1,2);
imagesc(axes1,dx_1,dy_1,qz_1);
caxis([min_u_ite,max_u_ite])
colormap jet
colorbar off
axis off
axes2 = axes(fig);
axes2.Position(1,3)=axes2.Position(1,3)/14*5;
axes2.Position(1,1)=axes1.Position(1,1);
axes2.Position(1,4)=axes2.Position(1,4)/2;
imagesc(axes2,dx_2,dy_2,qz_2)
caxis([min_u_ite,max_u_ite])
colorbar off
axis off
axes3 = axes(fig);
axes3.Position(1,3)=axes3.Position(1,3)/14*5;
axes3.Position(1,1)=axes2.Position(1,1)+axes2.Position(1,3);
axes3.Position(1,4)=axes3.Position(1,4)/2;
imagesc(axes3,dx_3,dy_3,qz_3)
caxis([min_u_ite,max_u_ite])
colorbar off
axis off
axes4 = axes(fig);
axes4.Position(1,3)=axes4.Position(1,3)/14*5;
axes4.Position(1,1)=axes3.Position(1,1);
axes4.Position(1,4)=axes4.Position(1,4)/2;
axes4.Position(1,2)=axes4.Position(1,4)+axes4.Position(1,2);
imagesc(axes4,dx_4,dy_4,qz_4)
caxis([min_u_ite,max_u_ite])
colorbar off
axis off
colormap jet
h=colorbar('manual');
set(h,'Position',[0.73,0.13,0.01,0.8])
set(h,'Fontsize',25)
set(gcf,'color','w')
axis off
set(gca,'FontSize',18);
savefile = strcat(strcat(savename,'-u-exact'),'.png');
saveas(gcf,strcat(savepath,savefile));
%---------------------%
% %% Grad_x of Exact solution over the entire domain
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fig=figure('NumberTitle','off','Name','grad_x Exact','Renderer', 'painters', 'Position', [0 0 700 500]);
% 
% load(strcat(path,'\gradu_exact_subR.mat'))
% u_NN_1 = gradu_ExactR(:,1);
% 
% load(strcat(path,'\gradu_exact_subB.mat'))
% u_NN_2 = gradu_ExactB(:,1);
% u_NN_1 = u_NN_1(10000:-1:1);
% u_NN_3 = u_NN_1(end:-1:10001);
% u_NN_2 = u_NN_2(10000:-1:1);
% u_NN_4 = u_NN_2(end:-1:10001);
% % mesh
% 
% 
% 
% Ft_1 = TriScatteredInterp(V_mesh_1(1,:)',V_mesh_1(2,:)',double(u_NN_1));
% qz_1 = Ft_1(qx_1,qy_1);
% 
% Ft_2 = TriScatteredInterp(V_mesh_3(1,:)',V_mesh_3(2,:)',double(u_NN_3));
% qz_3 = Ft_2(qx_3,qy_3);
% 
% Ft_3 = TriScatteredInterp(V_mesh_2(1,:)',V_mesh_2(2,:)',double(u_NN_2));
% qz_2 = Ft_3(qx_2,qy_2);
% 
% Ft_4 = TriScatteredInterp(V_mesh_4(1,:)',V_mesh_4(2,:)',double(u_NN_4));
% qz_4 = Ft_4(qx_4,qy_4);
% 
% % Ft = TriScatteredInterp(V_mesh_B(1,:)',V_mesh_B(2,:)',double(u_NN_2));
% % qz_B = Ft(qx_B,qy_B);
% 
% max_u_ite = double(max(max(u_NN_1),max(u_NN_2)));
% min_u_ite = double(min(min(u_NN_1),min(u_NN_2)));
% 
% %---------------------%
% axes1 = axes(fig);
% axes1.Position(1,3)=axes1.Position(1,3)/14*5;
% axes1.Position(1,4)=axes1.Position(1,4)/2;
% axes1.Position(1,2)=axes1.Position(1,4)+axes1.Position(1,2);
% imagesc(axes1,dx_1,dy_1,qz_1);
% caxis([min_u_ite,max_u_ite])
% colormap jet
% colorbar off
% axis off
% axes2 = axes(fig);
% axes2.Position(1,3)=axes2.Position(1,3)/14*5;
% axes2.Position(1,1)=axes1.Position(1,1);
% axes2.Position(1,4)=axes2.Position(1,4)/2;
% imagesc(axes2,dx_2,dy_2,qz_2)
% caxis([min_u_ite,max_u_ite])
% colorbar off
% axis off
% axes3 = axes(fig);
% axes3.Position(1,3)=axes3.Position(1,3)/14*5;
% axes3.Position(1,1)=axes2.Position(1,1)+axes2.Position(1,3);
% axes3.Position(1,4)=axes3.Position(1,4)/2;
% imagesc(axes3,dx_3,dy_3,qz_3)
% caxis([min_u_ite,max_u_ite])
% colorbar off
% axis off
% axes4 = axes(fig);
% axes4.Position(1,3)=axes4.Position(1,3)/14*5;
% axes4.Position(1,1)=axes3.Position(1,1);
% axes4.Position(1,4)=axes4.Position(1,4)/2;
% axes4.Position(1,2)=axes4.Position(1,4)+axes4.Position(1,2);
% imagesc(axes4,dx_4,dy_4,qz_4)
% caxis([min_u_ite,max_u_ite])
% colorbar off
% axis off
% colormap jet
% h=colorbar('manual');
% set(h,'Position',[0.73,0.13,0.01,0.8])
% set(h,'Fontsize',25)
% set(gcf,'color','w')
% axis off
% set(gca,'FontSize',18);
% saveas(gcf,strcat(savepath,'fig-DN-ex5-DN-PINNs-Gradu_x-exact.png'));
% %---------------------%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %% Grad_y of Exact solution over the entire domain
% fig=figure('NumberTitle','off','Name','grad_y','Renderer', 'painters', 'Position', [0 0 700 500]);
% 
% load(strcat(path,'\gradu_exact_subR.mat'))
% u_NN_1 = gradu_ExactR(:,2);
% load(strcat(path,'\gradu_exact_subB.mat'))
% u_NN_2 = gradu_ExactB(:,2);
% u_NN_1 = u_NN_1(10000:-1:1);
% u_NN_3 = u_NN_1(end:-1:10001);
% u_NN_2 = u_NN_2(10000:-1:1);
% u_NN_4 = u_NN_2(end:-1:10001);
% % mesh
% 
% 
% 
% 
% Ft_1 = TriScatteredInterp(V_mesh_1(1,:)',V_mesh_1(2,:)',double(u_NN_1));
% qz_1 = Ft_1(qx_1,qy_1);
% 
% Ft_2 = TriScatteredInterp(V_mesh_3(1,:)',V_mesh_3(2,:)',double(u_NN_3));
% qz_3 = Ft_2(qx_3,qy_3);
% 
% Ft_3 = TriScatteredInterp(V_mesh_2(1,:)',V_mesh_2(2,:)',double(u_NN_2));
% qz_2 = Ft_3(qx_2,qy_2);
% 
% Ft_4 = TriScatteredInterp(V_mesh_4(1,:)',V_mesh_4(2,:)',double(u_NN_4));
% qz_4 = Ft_4(qx_4,qy_4);
% 
% % Ft = TriScatteredInterp(V_mesh_B(1,:)',V_mesh_B(2,:)',double(u_NN_2));
% % qz_B = Ft(qx_B,qy_B);
% 
% max_u_ite = double(max(max(u_NN_1),max(u_NN_2)));
% min_u_ite = double(min(min(u_NN_1),min(u_NN_2)));
% 
% %---------------------%
% axes1 = axes(fig);
% axes1.Position(1,3)=axes1.Position(1,3)/14*5;
% axes1.Position(1,4)=axes1.Position(1,4)/2;
% axes1.Position(1,2)=axes1.Position(1,4)+axes1.Position(1,2);
% imagesc(axes1,dx_1,dy_1,qz_1);
% caxis([min_u_ite,max_u_ite])
% colormap jet
% colorbar off
% axis off
% axes2 = axes(fig);
% axes2.Position(1,3)=axes2.Position(1,3)/14*5;
% axes2.Position(1,1)=axes1.Position(1,1);
% axes2.Position(1,4)=axes2.Position(1,4)/2;
% imagesc(axes2,dx_2,dy_2,qz_2)
% caxis([min_u_ite,max_u_ite])
% colorbar off
% axis off
% axes3 = axes(fig);
% axes3.Position(1,3)=axes3.Position(1,3)/14*5;
% axes3.Position(1,1)=axes2.Position(1,1)+axes2.Position(1,3);
% axes3.Position(1,4)=axes3.Position(1,4)/2;
% imagesc(axes3,dx_3,dy_3,qz_3)
% caxis([min_u_ite,max_u_ite])
% colorbar off
% axis off
% axes4 = axes(fig);
% axes4.Position(1,3)=axes4.Position(1,3)/14*5;
% axes4.Position(1,1)=axes3.Position(1,1);
% axes4.Position(1,4)=axes4.Position(1,4)/2;
% axes4.Position(1,2)=axes4.Position(1,4)+axes4.Position(1,2);
% imagesc(axes4,dx_4,dy_4,qz_4)
% caxis([min_u_ite,max_u_ite])
% colorbar off
% axis off
% colormap jet
% h=colorbar('manual');
% set(h,'Position',[0.73,0.13,0.01,0.8])
% set(h,'Fontsize',25)
% set(gcf,'color','w')
% axis off
% set(gca,'FontSize',18);
% saveas(gcf,strcat(savepath,'fig-DN-ex5-DN-PINNs-Gradu_y-exact.png'));
% %---------------------%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Network predict solution final over entire domain
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
file = strcat(strcat('\u_NN_test_ite',index_1),'_sub1.mat');
load(strcat(path,file))
u_NN_1 = u_NN_sub1;
file = strcat(strcat('\u_NN_test_ite',index),'_sub2.mat');
load(strcat(path,file))
u_NN_2 = u_NN_sub2;
file = strcat(strcat('\u_NN_test_ite',index_1),'_sub3.mat');
load(strcat(path,file))
u_NN_3 = u_NN_sub3;
file = strcat(strcat('\u_NN_test_ite',index),'_sub4.mat');
load(strcat(path,file))
u_NN_4 = u_NN_sub4;


u_NN_1 = u_NN_1(10000:-1:1);
u_NN_2 = u_NN_2(10000:-1:1);
u_NN_3 = u_NN_3(10000:-1:1);
u_NN_4 = u_NN_4(10000:-1:1);
% mesh




Ft_1 = TriScatteredInterp(V_mesh_1(1,:)',V_mesh_1(2,:)',double(u_NN_1));
qz_1 = Ft_1(qx_1,qy_1);

Ft_2 = TriScatteredInterp(V_mesh_3(1,:)',V_mesh_3(2,:)',double(u_NN_3));
qz_3 = Ft_2(qx_3,qy_3);

Ft_3 = TriScatteredInterp(V_mesh_2(1,:)',V_mesh_2(2,:)',double(u_NN_2));
qz_2 = Ft_3(qx_2,qy_2);

Ft_4 = TriScatteredInterp(V_mesh_4(1,:)',V_mesh_4(2,:)',double(u_NN_4));
qz_4 = Ft_4(qx_4,qy_4);

% Ft = TriScatteredInterp(V_mesh_B(1,:)',V_mesh_B(2,:)',double(u_NN_2));
% qz_B = Ft(qx_B,qy_B);

max_u_ite = double(max(max(u_NN_1),max(u_NN_2)));
min_u_ite = double(min(min(u_NN_1),min(u_NN_2)));
fig=figure('NumberTitle','off','Name','predict solution','Renderer', 'painters', 'Position', [0 0 700 600]);

%---------------------%
axes1 = axes(fig);
axes1.Position(1,3) = axes1.Position(1,3)/7 * 5;
axes1.Position(1,2) = axes1.Position(1,4)/6;
axes1.Position(1,4) = axes1.Position(1,4)/6*5;
xticks([0  1]);
yticks([0  1]);
rectangle('position',[0 0 1 1] ,'LineWidth', 4);
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',25)
stringlabel = strcat(strcat('$\textnormal{Ite} = ',index),'$');
xlabel({stringlabel},'Interpreter','latex','FontSize',25)
axes3 = axes(fig);
axes3.Position(1,3)=axes3.Position(1,3)/14*5;
axes3.Position(1,1)=axes1.Position(1,1);
axes3.Position(1,2)=axes1.Position(1,2);
axes3.Position(1,4)=axes1.Position(1,4)/2;
imagesc(axes3,dx_2,dy_2,qz_2)
axis off;
caxis([min_u_ite,max_u_ite])

axes2 = axes(fig);
axes2.Position(1,3)=axes2.Position(1,3)/14*5;
axes2.Position(1,4)=axes1.Position(1,4)/2;
axes2.Position(1,2)=axes1.Position(1,2)+axes3.Position(1,4);


imagesc(axes2,dx_1,dy_1,qz_1);
caxis([min_u_ite,max_u_ite])
colormap jet
colorbar off
axis off
% axes3 = axes(fig);
% axes3.Position(1,3)=axes3.Position(1,3)/14*5;
% axes3.Position(1,1)=axes2.Position(1,1);
% axes3.Position(1,2)=axes1.Position(1,2);
% axes3.Position(1,4)=axes1.Position(1,4)/2;
% imagesc(axes3,dx_2,dy_2,qz_2)
% caxis([min_u_ite,max_u_ite])
% colorbar off
% axis off
axes4 = axes(fig);
axes4.Position(1,3)=axes4.Position(1,3)/14*5;
axes4.Position(1,1)=axes3.Position(1,1)+axes3.Position(1,3);
axes4.Position(1,4)=axes1.Position(1,4)/2;
axes4.Position(1,2)=axes1.Position(1,2);
imagesc(axes4,dx_3,dy_3,qz_3)
caxis([min_u_ite,max_u_ite])
colorbar off
axis off
axes5 = axes(fig);
axes5.Position(1,3)=axes5.Position(1,3)/14*5;
axes5.Position(1,1)=axes4.Position(1,1);
axes5.Position(1,4)=axes1.Position(1,4)/2;
axes5.Position(1,2)=axes1.Position(1,2)+axes4.Position(1,4);
imagesc(axes5,dx_4,dy_4,qz_4)
caxis([min_u_ite,max_u_ite])
colorbar off
axis off
colormap jet
h=colorbar('manual');
set(h,'Position',[0.73,0.13,0.01,0.7])
set(h,'Fontsize',25)
set(gcf,'color','w')
axis off
set(gca,'FontSize',18);
savefile = strcat(strcat(strcat(savename,strcat(strcat(algorithm,'u_NN-'),'ite-')),index),'.png');
saveas(gcf,strcat(savepath,savefile));
%---------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% pointwise err final over entire domain
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
file = strcat(strcat('\err_test_ite',index_1),'_sub1.mat');
load(strcat(path,file))
err_1 = pointerr1;
file = strcat(strcat('\err_test_ite',index),'_sub2.mat');
load(strcat(path,file))
err_2 = pointerr2;
file = strcat(strcat('\err_test_ite',index_1),'_sub3.mat');
load(strcat(path,file))
err_3 = pointerr3;
file = strcat(strcat('\err_test_ite',index),'_sub4.mat');
load(strcat(path,file))
err_4 = pointerr4;


err_1 = err_1(10000:-1:1);
err_2 = err_2(10000:-1:1);
err_3 = err_3(10000:-1:1);
err_4 = err_4(10000:-1:1);
% mesh




Ft_1 = TriScatteredInterp(V_mesh_1(1,:)',V_mesh_1(2,:)',double(err_1));
qz_1 = Ft_1(qx_1,qy_1);

Ft_2 = TriScatteredInterp(V_mesh_3(1,:)',V_mesh_3(2,:)',double(err_3));
qz_3 = Ft_2(qx_3,qy_3);

Ft_3 = TriScatteredInterp(V_mesh_2(1,:)',V_mesh_2(2,:)',double(err_2));
qz_2 = Ft_3(qx_2,qy_2);

Ft_4 = TriScatteredInterp(V_mesh_4(1,:)',V_mesh_4(2,:)',double(err_4));
qz_4 = Ft_4(qx_4,qy_4);

% Ft = TriScatteredInterp(V_mesh_B(1,:)',V_mesh_B(2,:)',double(u_NN_2));
% qz_B = Ft(qx_B,qy_B);

max_u_ite = double(max([max(err_1),max(err_2),max(err_3),max(err_4)]));
min_u_ite = double(min([min(err_1),min(err_2),min(err_3),min(err_4)]));
fig=figure('NumberTitle','off','Name','pterr L2','Renderer', 'painters', 'Position', [0 0 700 600]);

%---------------------%
axes1 = axes(fig);
axes1.Position(1,3) = axes1.Position(1,3)/7 * 5;
axes1.Position(1,2) = axes1.Position(1,4)/6;
axes1.Position(1,4) = axes1.Position(1,4)/6*5;
xticks([0  1]);
yticks([0  1]);
rectangle('position',[0 0 1 1] ,'LineWidth', 4);
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',25)
stringlabel = strcat(strcat('$\textnormal{Ite} = ',index),'$');
xlabel({stringlabel},'Interpreter','latex','FontSize',25)
axes3 = axes(fig);
axes3.Position(1,3)=axes3.Position(1,3)/14*5;
axes3.Position(1,1)=axes1.Position(1,1);
axes3.Position(1,2)=axes1.Position(1,2);
axes3.Position(1,4)=axes1.Position(1,4)/2;
imagesc(axes3,dx_2,dy_2,qz_2)
axis off;
caxis([min_u_ite,max_u_ite])

axes2 = axes(fig);
axes2.Position(1,3)=axes2.Position(1,3)/14*5;
axes2.Position(1,4)=axes1.Position(1,4)/2;
axes2.Position(1,2)=axes1.Position(1,2)+axes3.Position(1,4);


imagesc(axes2,dx_1,dy_1,qz_1);
caxis([min_u_ite,max_u_ite])
colormap jet
colorbar off
axis off
% axes3 = axes(fig);
% axes3.Position(1,3)=axes3.Position(1,3)/14*5;
% axes3.Position(1,1)=axes2.Position(1,1);
% axes3.Position(1,2)=axes1.Position(1,2);
% axes3.Position(1,4)=axes1.Position(1,4)/2;
% imagesc(axes3,dx_2,dy_2,qz_2)
% caxis([min_u_ite,max_u_ite])
% colorbar off
% axis off
axes4 = axes(fig);
axes4.Position(1,3)=axes4.Position(1,3)/14*5;
axes4.Position(1,1)=axes3.Position(1,1)+axes3.Position(1,3);
axes4.Position(1,4)=axes1.Position(1,4)/2;
axes4.Position(1,2)=axes1.Position(1,2);
imagesc(axes4,dx_3,dy_3,qz_3)
caxis([min_u_ite,max_u_ite])
colorbar off
axis off
axes5 = axes(fig);
axes5.Position(1,3)=axes5.Position(1,3)/14*5;
axes5.Position(1,1)=axes4.Position(1,1);
axes5.Position(1,4)=axes1.Position(1,4)/2;
axes5.Position(1,2)=axes1.Position(1,2)+axes4.Position(1,4);
imagesc(axes5,dx_4,dy_4,qz_4)
caxis([min_u_ite,max_u_ite])
colorbar off
axis off
colormap jet
h=colorbar('manual');
set(h,'Position',[0.73,0.13,0.01,0.7])
set(h,'Fontsize',25)
set(gcf,'color','w')
axis off
set(gca,'FontSize',18);
savefile = strcat(strcat(strcat(savename,strcat(strcat(algorithm,'pterr-'),'ite-')),index),'.png');
saveas(gcf,strcat(savepath,savefile));
%---------------------%


%% pterr gradient x  over the entire domain
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load(strcat(path,'\gradu_exact_sub1.mat'))
load(strcat(path,'\gradu_exact_sub2.mat'))
load(strcat(path,'\gradu_exact_sub3.mat'))
load(strcat(path,'\gradu_exact_sub4.mat'))


load(strcat(path,strcat('\gradu_NN_test_ite',index),'_sub1.mat'))
load(strcat(path,strcat('\gradu_NN_test_ite',index),'_sub2.mat'))
load(strcat(path,strcat('\gradu_NN_test_ite',index),'_sub3.mat'))
load(strcat(path,strcat('\gradu_NN_test_ite',index),'_sub4.mat'))

err_1 = gradu_Exact1(:,1) - grad_u_test1(:,1);
err_2 = gradu_Exact2(:,1) - grad_u_test2(:,1);
err_3 = gradu_Exact3(:,1) - grad_u_test3(:,1);
err_4 = gradu_Exact4(:,1) - grad_u_test4(:,1);



err_1 = err_1(10000:-1:1);
err_2 = err_2(10000:-1:1);
err_3 = err_3(10000:-1:1);
err_4 = err_4(10000:-1:1);



Ft_1 = TriScatteredInterp(V_mesh_1(1,:)',V_mesh_1(2,:)',double(err_1));
qz_1 = Ft_1(qx_1,qy_1);

Ft_2 = TriScatteredInterp(V_mesh_3(1,:)',V_mesh_3(2,:)',double(err_3));
qz_3 = Ft_2(qx_3,qy_3);

Ft_3 = TriScatteredInterp(V_mesh_2(1,:)',V_mesh_2(2,:)',double(err_2));
qz_2 = Ft_3(qx_2,qy_2);

Ft_4 = TriScatteredInterp(V_mesh_4(1,:)',V_mesh_4(2,:)',double(err_4));
qz_4 = Ft_4(qx_4,qy_4);

% Ft = TriScatteredInterp(V_mesh_B(1,:)',V_mesh_B(2,:)',double(u_NN_2));
% qz_B = Ft(qx_B,qy_B);

max_u_ite = double(max([max(err_1),max(err_2),max(err_3),max(err_4)]));
min_u_ite = double(min([min(err_1),min(err_2),min(err_3),min(err_4)]));
fig=figure('NumberTitle','off','Name','pterr dx','Renderer', 'painters', 'Position', [0 0 700 600]);

%---------------------%
axes1 = axes(fig);
axes1.Position(1,3) = axes1.Position(1,3)/7 * 5;
axes1.Position(1,2) = axes1.Position(1,4)/6;
axes1.Position(1,4) = axes1.Position(1,4)/6*5;
xticks([0  1]);
yticks([0  1]);
rectangle('position',[0 0 1 1] ,'LineWidth', 4);
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',25)
stringlabel = strcat(strcat('$\textnormal{Ite} = ',index),'$');
xlabel({stringlabel},'Interpreter','latex','FontSize',25)
axes3 = axes(fig);
axes3.Position(1,3)=axes3.Position(1,3)/14*5;
axes3.Position(1,1)=axes1.Position(1,1);
axes3.Position(1,2)=axes1.Position(1,2);
axes3.Position(1,4)=axes1.Position(1,4)/2;
imagesc(axes3,dx_2,dy_2,qz_2)
axis off;
caxis([min_u_ite,max_u_ite])

axes2 = axes(fig);
axes2.Position(1,3)=axes2.Position(1,3)/14*5;
axes2.Position(1,4)=axes1.Position(1,4)/2;
axes2.Position(1,2)=axes1.Position(1,2)+axes3.Position(1,4);


imagesc(axes2,dx_1,dy_1,qz_1);
caxis([min_u_ite,max_u_ite])
colormap jet
colorbar off
axis off
% axes3 = axes(fig);
% axes3.Position(1,3)=axes3.Position(1,3)/14*5;
% axes3.Position(1,1)=axes2.Position(1,1);
% axes3.Position(1,2)=axes1.Position(1,2);
% axes3.Position(1,4)=axes1.Position(1,4)/2;
% imagesc(axes3,dx_2,dy_2,qz_2)
% caxis([min_u_ite,max_u_ite])
% colorbar off
% axis off
axes4 = axes(fig);
axes4.Position(1,3)=axes4.Position(1,3)/14*5;
axes4.Position(1,1)=axes3.Position(1,1)+axes3.Position(1,3);
axes4.Position(1,4)=axes1.Position(1,4)/2;
axes4.Position(1,2)=axes1.Position(1,2);
imagesc(axes4,dx_3,dy_3,qz_3)
caxis([min_u_ite,max_u_ite])
colorbar off
axis off
axes5 = axes(fig);
axes5.Position(1,3)=axes5.Position(1,3)/14*5;
axes5.Position(1,1)=axes4.Position(1,1);
axes5.Position(1,4)=axes1.Position(1,4)/2;
axes5.Position(1,2)=axes1.Position(1,2)+axes4.Position(1,4);
imagesc(axes5,dx_4,dy_4,qz_4)
caxis([min_u_ite,max_u_ite])
colorbar off
axis off
colormap jet
h=colorbar('manual');
set(h,'Position',[0.73,0.13,0.01,0.7])
set(h,'Fontsize',25)
set(gcf,'color','w')
axis off
set(gca,'FontSize',18);
savefile = strcat(strcat(strcat(savename,strcat(strcat(algorithm,'pterr-dx-'),'ite-')),index),'.png');
saveas(gcf,strcat(savepath,savefile));
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