% plot figures one by one on 4 sub domain

format short e
close all
path = 'D:\博士研究生\研二\Prj22-DDLM-V1-Code\main-DNLM-PINN\Codes\Results\2_4Prob-2D\DN-PINNs\G1e_2-N2e4-baseline\simulation-test-1';
savepath = 'D:\博士研究生\highcontrast\Figures\2_4Prob-2D\DN-PINNs\simulation-1\';
savename = 'fig-DN-ex3-';
algorithm = 'DN-PINNs-';

if(exist(savepath,'dir')~=7)
    mkdir(savepath);
end
index = '11';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% problem setting
num_pts = 100;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% exact solution over entire domain
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig=figure('NumberTitle','off','Name','Exact Solution','Renderer', 'painters', 'Position', [0 0 700 500]);


load(strcat(path,'\u_exact_subR.mat'))
u_NN_R = u_exR;

load(strcat(path,'\u_exact_subB.mat'))
u_NN_B = u_exB;
u_NN_R_1 = u_NN_R(10000:-1:1);
u_NN_R_2 = u_NN_R(end:-1:10001);
u_NN_B_1 = u_NN_B(10000:-1:1);
u_NN_B_2 = u_NN_B(end:-1:10001);
% mesh
[V_mesh_R_1, qx_R_1, qy_R_1, dx_R_1, dy_R_1] = generate_mesh(0, 0.5, 0.5, 1, num_pts);
[V_mesh_R_2, qx_R_2, qy_R_2, dx_R_2, dy_R_2] = generate_mesh(0.5, 1, 0, 0.5, num_pts);

[V_mesh_B_1, qx_B_1, qy_B_1, dx_B_1, dy_B_1] = generate_mesh(0, 0.5, 0, 0.5, num_pts);
[V_mesh_B_2, qx_B_2, qy_B_2, dx_B_2, dy_B_2] = generate_mesh(0.5, 1, 0.5, 1, num_pts);

dx_R_1 = dx_R_1(end:-1:1);
dx_R_2 = dx_R_2(end:-1:1);
dx_B_1 = dx_B_1(end:-1:1);
dx_B_2 = dx_B_2(end:-1:1);


Ft_1 = TriScatteredInterp(V_mesh_R_1(1,:)',V_mesh_R_1(2,:)',double(u_NN_R_1));
qz_R_1 = Ft_1(qx_R_1,qy_R_1);

Ft_2 = TriScatteredInterp(V_mesh_R_2(1,:)',V_mesh_R_2(2,:)',double(u_NN_R_2));
qz_R_2 = Ft_2(qx_R_2,qy_R_2);

Ft_3 = TriScatteredInterp(V_mesh_B_1(1,:)',V_mesh_B_1(2,:)',double(u_NN_B_1));
qz_B_1 = Ft_3(qx_B_1,qy_B_1);

Ft_4 = TriScatteredInterp(V_mesh_B_2(1,:)',V_mesh_B_2(2,:)',double(u_NN_B_2));
qz_B_2 = Ft_4(qx_B_2,qy_B_2);

% Ft = TriScatteredInterp(V_mesh_B(1,:)',V_mesh_B(2,:)',double(u_NN_B));
% qz_B = Ft(qx_B,qy_B);

max_u_ite = double(max(max(u_NN_R),max(u_NN_B)));
min_u_ite = double(min(min(u_NN_R),min(u_NN_B)));

%---------------------%
axes1 = axes(fig);
axes1.Position(1,3)=axes1.Position(1,3)/14*5;
axes1.Position(1,4)=axes1.Position(1,4)/2;
axes1.Position(1,2)=axes1.Position(1,4)+axes1.Position(1,2);
imagesc(axes1,dx_R_1,dy_R_1,qz_R_1);
caxis([min_u_ite,max_u_ite])
colormap jet
colorbar off
axis off
axes2 = axes(fig);
axes2.Position(1,3)=axes2.Position(1,3)/14*5;
axes2.Position(1,1)=axes1.Position(1,1);
axes2.Position(1,4)=axes2.Position(1,4)/2;
imagesc(axes2,dx_B_1,dy_B_1,qz_B_1)
caxis([min_u_ite,max_u_ite])
colorbar off
axis off
axes3 = axes(fig);
axes3.Position(1,3)=axes3.Position(1,3)/14*5;
axes3.Position(1,1)=axes2.Position(1,1)+axes2.Position(1,3);
axes3.Position(1,4)=axes3.Position(1,4)/2;
imagesc(axes3,dx_R_2,dy_R_2,qz_R_2)
caxis([min_u_ite,max_u_ite])
colorbar off
axis off
axes4 = axes(fig);
axes4.Position(1,3)=axes4.Position(1,3)/14*5;
axes4.Position(1,1)=axes3.Position(1,1);
axes4.Position(1,4)=axes4.Position(1,4)/2;
axes4.Position(1,2)=axes4.Position(1,4)+axes4.Position(1,2);
imagesc(axes4,dx_B_2,dy_B_2,qz_B_2)
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
%% Grad_x of Exact solution over the entire domain
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig=figure('NumberTitle','off','Name','grad_x Exact','Renderer', 'painters', 'Position', [0 0 700 500]);

load(strcat(path,'\gradu_exact_subR.mat'))
u_NN_R = gradu_ExactR(:,1);

load(strcat(path,'\gradu_exact_subB.mat'))
u_NN_B = gradu_ExactB(:,1);
u_NN_R_1 = u_NN_R(10000:-1:1);
u_NN_R_2 = u_NN_R(end:-1:10001);
u_NN_B_1 = u_NN_B(10000:-1:1);
u_NN_B_2 = u_NN_B(end:-1:10001);
% mesh



Ft_1 = TriScatteredInterp(V_mesh_R_1(1,:)',V_mesh_R_1(2,:)',double(u_NN_R_1));
qz_R_1 = Ft_1(qx_R_1,qy_R_1);

Ft_2 = TriScatteredInterp(V_mesh_R_2(1,:)',V_mesh_R_2(2,:)',double(u_NN_R_2));
qz_R_2 = Ft_2(qx_R_2,qy_R_2);

Ft_3 = TriScatteredInterp(V_mesh_B_1(1,:)',V_mesh_B_1(2,:)',double(u_NN_B_1));
qz_B_1 = Ft_3(qx_B_1,qy_B_1);

Ft_4 = TriScatteredInterp(V_mesh_B_2(1,:)',V_mesh_B_2(2,:)',double(u_NN_B_2));
qz_B_2 = Ft_4(qx_B_2,qy_B_2);

% Ft = TriScatteredInterp(V_mesh_B(1,:)',V_mesh_B(2,:)',double(u_NN_B));
% qz_B = Ft(qx_B,qy_B);

max_u_ite = double(max(max(u_NN_R),max(u_NN_B)));
min_u_ite = double(min(min(u_NN_R),min(u_NN_B)));

%---------------------%
axes1 = axes(fig);
axes1.Position(1,3)=axes1.Position(1,3)/14*5;
axes1.Position(1,4)=axes1.Position(1,4)/2;
axes1.Position(1,2)=axes1.Position(1,4)+axes1.Position(1,2);
imagesc(axes1,dx_R_1,dy_R_1,qz_R_1);
caxis([min_u_ite,max_u_ite])
colormap jet
colorbar off
axis off
axes2 = axes(fig);
axes2.Position(1,3)=axes2.Position(1,3)/14*5;
axes2.Position(1,1)=axes1.Position(1,1);
axes2.Position(1,4)=axes2.Position(1,4)/2;
imagesc(axes2,dx_B_1,dy_B_1,qz_B_1)
caxis([min_u_ite,max_u_ite])
colorbar off
axis off
axes3 = axes(fig);
axes3.Position(1,3)=axes3.Position(1,3)/14*5;
axes3.Position(1,1)=axes2.Position(1,1)+axes2.Position(1,3);
axes3.Position(1,4)=axes3.Position(1,4)/2;
imagesc(axes3,dx_R_2,dy_R_2,qz_R_2)
caxis([min_u_ite,max_u_ite])
colorbar off
axis off
axes4 = axes(fig);
axes4.Position(1,3)=axes4.Position(1,3)/14*5;
axes4.Position(1,1)=axes3.Position(1,1);
axes4.Position(1,4)=axes4.Position(1,4)/2;
axes4.Position(1,2)=axes4.Position(1,4)+axes4.Position(1,2);
imagesc(axes4,dx_B_2,dy_B_2,qz_B_2)
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
saveas(gcf,strcat(savepath,'fig-DN-ex5-DN-PINNs-Gradu_x-exact.png'));
%---------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Grad_y of Exact solution over the entire domain
fig=figure('NumberTitle','off','Name','grad_y','Renderer', 'painters', 'Position', [0 0 700 500]);

load(strcat(path,'\gradu_exact_subR.mat'))
u_NN_R = gradu_ExactR(:,2);
load(strcat(path,'\gradu_exact_subB.mat'))
u_NN_B = gradu_ExactB(:,2);
u_NN_R_1 = u_NN_R(10000:-1:1);
u_NN_R_2 = u_NN_R(end:-1:10001);
u_NN_B_1 = u_NN_B(10000:-1:1);
u_NN_B_2 = u_NN_B(end:-1:10001);
% mesh




Ft_1 = TriScatteredInterp(V_mesh_R_1(1,:)',V_mesh_R_1(2,:)',double(u_NN_R_1));
qz_R_1 = Ft_1(qx_R_1,qy_R_1);

Ft_2 = TriScatteredInterp(V_mesh_R_2(1,:)',V_mesh_R_2(2,:)',double(u_NN_R_2));
qz_R_2 = Ft_2(qx_R_2,qy_R_2);

Ft_3 = TriScatteredInterp(V_mesh_B_1(1,:)',V_mesh_B_1(2,:)',double(u_NN_B_1));
qz_B_1 = Ft_3(qx_B_1,qy_B_1);

Ft_4 = TriScatteredInterp(V_mesh_B_2(1,:)',V_mesh_B_2(2,:)',double(u_NN_B_2));
qz_B_2 = Ft_4(qx_B_2,qy_B_2);

% Ft = TriScatteredInterp(V_mesh_B(1,:)',V_mesh_B(2,:)',double(u_NN_B));
% qz_B = Ft(qx_B,qy_B);

max_u_ite = double(max(max(u_NN_R),max(u_NN_B)));
min_u_ite = double(min(min(u_NN_R),min(u_NN_B)));

%---------------------%
axes1 = axes(fig);
axes1.Position(1,3)=axes1.Position(1,3)/14*5;
axes1.Position(1,4)=axes1.Position(1,4)/2;
axes1.Position(1,2)=axes1.Position(1,4)+axes1.Position(1,2);
imagesc(axes1,dx_R_1,dy_R_1,qz_R_1);
caxis([min_u_ite,max_u_ite])
colormap jet
colorbar off
axis off
axes2 = axes(fig);
axes2.Position(1,3)=axes2.Position(1,3)/14*5;
axes2.Position(1,1)=axes1.Position(1,1);
axes2.Position(1,4)=axes2.Position(1,4)/2;
imagesc(axes2,dx_B_1,dy_B_1,qz_B_1)
caxis([min_u_ite,max_u_ite])
colorbar off
axis off
axes3 = axes(fig);
axes3.Position(1,3)=axes3.Position(1,3)/14*5;
axes3.Position(1,1)=axes2.Position(1,1)+axes2.Position(1,3);
axes3.Position(1,4)=axes3.Position(1,4)/2;
imagesc(axes3,dx_R_2,dy_R_2,qz_R_2)
caxis([min_u_ite,max_u_ite])
colorbar off
axis off
axes4 = axes(fig);
axes4.Position(1,3)=axes4.Position(1,3)/14*5;
axes4.Position(1,1)=axes3.Position(1,1);
axes4.Position(1,4)=axes4.Position(1,4)/2;
axes4.Position(1,2)=axes4.Position(1,4)+axes4.Position(1,2);
imagesc(axes4,dx_B_2,dy_B_2,qz_B_2)
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
saveas(gcf,strcat(savepath,'fig-DN-ex5-DN-PINNs-Gradu_y-exact.png'));
%---------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Network predict solution final over entire domain
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
file = strcat(strcat('\u_NN_test_ite',index),'_subR.mat');
load(strcat(path,file))

u_NN_R = u_NN_subR;

file = strcat(strcat('\u_NN_test_ite',index),'_subB.mat');
load(strcat(path,file))

u_NN_B = u_NN_subB;
u_NN_R_1 = u_NN_R(10000:-1:1);
u_NN_R_2 = u_NN_R(end:-1:10001);
u_NN_B_1 = u_NN_B(10000:-1:1);
u_NN_B_2 = u_NN_B(end:-1:10001);
% mesh




Ft_1 = TriScatteredInterp(V_mesh_R_1(1,:)',V_mesh_R_1(2,:)',double(u_NN_R_1));
qz_R_1 = Ft_1(qx_R_1,qy_R_1);

Ft_2 = TriScatteredInterp(V_mesh_R_2(1,:)',V_mesh_R_2(2,:)',double(u_NN_R_2));
qz_R_2 = Ft_2(qx_R_2,qy_R_2);

Ft_3 = TriScatteredInterp(V_mesh_B_1(1,:)',V_mesh_B_1(2,:)',double(u_NN_B_1));
qz_B_1 = Ft_3(qx_B_1,qy_B_1);

Ft_4 = TriScatteredInterp(V_mesh_B_2(1,:)',V_mesh_B_2(2,:)',double(u_NN_B_2));
qz_B_2 = Ft_4(qx_B_2,qy_B_2);

% Ft = TriScatteredInterp(V_mesh_B(1,:)',V_mesh_B(2,:)',double(u_NN_B));
% qz_B = Ft(qx_B,qy_B);

max_u_ite = double(max(max(u_NN_R),max(u_NN_B)));
min_u_ite = double(min(min(u_NN_R),min(u_NN_B)));
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
imagesc(axes3,dx_B_1,dy_B_1,qz_B_1)
axis off;
caxis([min_u_ite,max_u_ite])

axes2 = axes(fig);
axes2.Position(1,3)=axes2.Position(1,3)/14*5;
axes2.Position(1,4)=axes1.Position(1,4)/2;
axes2.Position(1,2)=axes1.Position(1,2)+axes3.Position(1,4);


imagesc(axes2,dx_R_1,dy_R_1,qz_R_1);
caxis([min_u_ite,max_u_ite])
colormap jet
colorbar off
axis off
% axes3 = axes(fig);
% axes3.Position(1,3)=axes3.Position(1,3)/14*5;
% axes3.Position(1,1)=axes2.Position(1,1);
% axes3.Position(1,2)=axes1.Position(1,2);
% axes3.Position(1,4)=axes1.Position(1,4)/2;
% imagesc(axes3,dx_B_1,dy_B_1,qz_B_1)
% caxis([min_u_ite,max_u_ite])
% colorbar off
% axis off
axes4 = axes(fig);
axes4.Position(1,3)=axes4.Position(1,3)/14*5;
axes4.Position(1,1)=axes3.Position(1,1)+axes3.Position(1,3);
axes4.Position(1,4)=axes1.Position(1,4)/2;
axes4.Position(1,2)=axes1.Position(1,2);
imagesc(axes4,dx_R_2,dy_R_2,qz_R_2)
caxis([min_u_ite,max_u_ite])
colorbar off
axis off
axes5 = axes(fig);
axes5.Position(1,3)=axes5.Position(1,3)/14*5;
axes5.Position(1,1)=axes4.Position(1,1);
axes5.Position(1,4)=axes1.Position(1,4)/2;
axes5.Position(1,2)=axes1.Position(1,2)+axes4.Position(1,4);
imagesc(axes5,dx_B_2,dy_B_2,qz_B_2)
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
savefile = strcat(strcat(strcat(savename,strcat(strcat(algorithm,'u-NN-'),'ite-')),index),'.png');
saveas(gcf,strcat(savepath,savefile));
%---------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% pointwise err final over entire domain
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
file = strcat(strcat('\err_test_ite',index),'_subR.mat');
load(strcat(path,file))
err_R = abs(pointerrR);
err_R_1 = err_R(10001:end);
err_R_2 = err_R(1:10000);
file = strcat(strcat('\err_test_ite',index),'_subB.mat');
load(strcat(path,file))
err_B = abs(pointerrB);
err_R_1 = err_R(10000:-1:1);
err_R_2 = err_R(end:-1:10001);
err_B_1 = err_B(10000:-1:1);
err_B_2 = err_B(end:-1:10001);
% mesh




Ft_1 = TriScatteredInterp(V_mesh_R_1(1,:)',V_mesh_R_1(2,:)',double(err_R_1));
qz_R_1 = Ft_1(qx_R_1,qy_R_1);

Ft_2 = TriScatteredInterp(V_mesh_R_2(1,:)',V_mesh_R_2(2,:)',double(err_R_2));
qz_R_2 = Ft_2(qx_R_2,qy_R_2);

Ft_3 = TriScatteredInterp(V_mesh_B_1(1,:)',V_mesh_B_1(2,:)',double(err_B_1));
qz_B_1 = Ft_3(qx_B_1,qy_B_1);

Ft_4 = TriScatteredInterp(V_mesh_B_2(1,:)',V_mesh_B_2(2,:)',double(err_B_2));
qz_B_2 = Ft_4(qx_B_2,qy_B_2);

% Ft = TriScatteredInterp(V_mesh_B(1,:)',V_mesh_B(2,:)',double(u_NN_B));
% qz_B = Ft(qx_B,qy_B);

max_u_ite = double(max(max(err_R),max(err_B)));
min_u_ite = double(min(min(err_R),min(err_B)));
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
imagesc(axes3,dx_B_1,dy_B_1,qz_B_1)
axis off;
caxis([min_u_ite,max_u_ite])

axes2 = axes(fig);
axes2.Position(1,3)=axes2.Position(1,3)/14*5;
axes2.Position(1,4)=axes1.Position(1,4)/2;
axes2.Position(1,2)=axes1.Position(1,2)+axes3.Position(1,4);


imagesc(axes2,dx_R_1,dy_R_1,qz_R_1);
caxis([min_u_ite,max_u_ite])
colormap jet
colorbar off
axis off
% axes3 = axes(fig);
% axes3.Position(1,3)=axes3.Position(1,3)/14*5;
% axes3.Position(1,1)=axes2.Position(1,1);
% axes3.Position(1,2)=axes1.Position(1,2);
% axes3.Position(1,4)=axes1.Position(1,4)/2;
% imagesc(axes3,dx_B_1,dy_B_1,qz_B_1)
% caxis([min_u_ite,max_u_ite])
% colorbar off
% axis off
axes4 = axes(fig);
axes4.Position(1,3)=axes4.Position(1,3)/14*5;
axes4.Position(1,1)=axes3.Position(1,1)+axes3.Position(1,3);
axes4.Position(1,4)=axes1.Position(1,4)/2;
axes4.Position(1,2)=axes1.Position(1,2);
imagesc(axes4,dx_R_2,dy_R_2,qz_R_2)
caxis([min_u_ite,max_u_ite])
colorbar off
axis off
axes5 = axes(fig);
axes5.Position(1,3)=axes5.Position(1,3)/14*5;
axes5.Position(1,1)=axes4.Position(1,1);
axes5.Position(1,4)=axes1.Position(1,4)/2;
axes5.Position(1,2)=axes1.Position(1,2)+axes4.Position(1,4);
imagesc(axes5,dx_B_2,dy_B_2,qz_B_2)
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


%% pterr H1  over the entire domain
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load(strcat(path,'\gradu_exact_subR.mat'))


load(strcat(path,'\gradu_exact_subB.mat'))
load(strcat(path,strcat('\gradu_NN_test_ite',index),'_subB.mat'))
load(strcat(path,strcat('\gradu_NN_test_ite',index),'_subR.mat'))
error_H1_R = ((gradu_ExactR(:,1)-grad_u_testR(:,1)).*(gradu_ExactR(:,1)-grad_u_testR(:,1))).^0.5;
error_H1_B = ((gradu_ExactB(:,1)-grad_u_testB(:,1)).*(gradu_ExactB(:,1)-grad_u_testB(:,1))).^0.5;
error_H1_R_1 = error_H1_R(10000:-1:1);
error_H1_R_2 = error_H1_R(end:-1:10001);
error_H1_B_1 = error_H1_B(10000:-1:1);
error_H1_B_2= error_H1_B(end:-1:10001);
% mesh



Ft_1 = TriScatteredInterp(V_mesh_R_1(1,:)',V_mesh_R_1(2,:)',double(error_H1_R_1));
qz_R_1 = Ft_1(qx_R_1,qy_R_1);

Ft_2 = TriScatteredInterp(V_mesh_R_2(1,:)',V_mesh_R_2(2,:)',double(error_H1_R_2));
qz_R_2 = Ft_2(qx_R_2,qy_R_2);

Ft_3 = TriScatteredInterp(V_mesh_B_1(1,:)',V_mesh_B_1(2,:)',double(error_H1_B_1));
qz_B_1 = Ft_3(qx_B_1,qy_B_1);

Ft_4 = TriScatteredInterp(V_mesh_B_2(1,:)',V_mesh_B_2(2,:)',double(error_H1_B_2));
qz_B_2 = Ft_4(qx_B_2,qy_B_2);

% Ft = TriScatteredInterp(V_mesh_B(1,:)',V_mesh_B(2,:)',double(u_NN_B));
% qz_B = Ft(qx_B,qy_B);

max_u_ite = double(max(max(error_H1_B_1),max(error_H1_R_1 )));
min_u_ite = double(min(min(error_H1_R_1 ),min(error_H1_B_1 )));
fig=figure('NumberTitle','off','Name','pterr H1','Renderer', 'painters', 'Position', [0 0 700 600]);

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
imagesc(axes3,dx_B_1,dy_B_1,qz_B_1)
axis off;
caxis([min_u_ite,max_u_ite])

axes2 = axes(fig);
axes2.Position(1,3)=axes2.Position(1,3)/14*5;
axes2.Position(1,4)=axes1.Position(1,4)/2;
axes2.Position(1,2)=axes1.Position(1,2)+axes3.Position(1,4);


imagesc(axes2,dx_R_1,dy_R_1,qz_R_1);
caxis([min_u_ite,max_u_ite])
colormap jet
colorbar off
axis off
% axes3 = axes(fig);
% axes3.Position(1,3)=axes3.Position(1,3)/14*5;
% axes3.Position(1,1)=axes2.Position(1,1);
% axes3.Position(1,2)=axes1.Position(1,2);
% axes3.Position(1,4)=axes1.Position(1,4)/2;
% imagesc(axes3,dx_B_1,dy_B_1,qz_B_1)
% caxis([min_u_ite,max_u_ite])
% colorbar off
% axis off
axes4 = axes(fig);
axes4.Position(1,3)=axes4.Position(1,3)/14*5;
axes4.Position(1,1)=axes3.Position(1,1)+axes3.Position(1,3);
axes4.Position(1,4)=axes1.Position(1,4)/2;
axes4.Position(1,2)=axes1.Position(1,2);
imagesc(axes4,dx_R_2,dy_R_2,qz_R_2)
caxis([min_u_ite,max_u_ite])
colorbar off
axis off
axes5 = axes(fig);
axes5.Position(1,3)=axes5.Position(1,3)/14*5;
axes5.Position(1,1)=axes4.Position(1,1);
axes5.Position(1,4)=axes1.Position(1,4)/2;
axes5.Position(1,2)=axes1.Position(1,2)+axes4.Position(1,4);
imagesc(axes5,dx_B_2,dy_B_2,qz_B_2)
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
savefile = strcat(strcat(strcat(savename,strcat(strcat(algorithm,'pterr-H1-'),'ite-')),index),'.png');
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