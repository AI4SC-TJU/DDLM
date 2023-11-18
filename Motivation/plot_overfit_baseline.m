%plot_DDLM_overfit.m
% plot figures one by one

format short e
close all
path = 'D:\博士研究生\研二\Prj22-DDLM-V1-Code\main-DNLM-PINN\Codes\Results\Overfit\Dirichlet\2D-squaure\epoch5k\simulation-3';
savepath = 'D:\博士研究生\研二\Prj22-DDLM-V1-Code\main-DNLM-PINN\Figures\Overfit\Dirichlet\2D-squaure\simulation-1e4-2\';
if(exist(savepath,'dir')~=7)
    mkdir(savepath);
end
boundary = '-Dirichlet-overfit';
algorithm = 'PINN-e1w';
index = '1';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% problem setting
num_pts = 100;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% exact solution over entire domain
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% mesh
[V_mesh, qx, qy, dx, dy] = generate_mesh(0, 0.5, 0, 1, num_pts);
% exact solution
U = @(x,y) (sin(2*pi.*x)).* (cos(2*pi.*y) - 1); 
U_x = @(x,y) 2*pi*cos(2*pi*x).*(cos(2*pi.*y)-1);
U_y = @(x,y) -2*pi*sin(2*pi*x).*sin(2*pi.*y);
u_exact = U(V_mesh(1,:),V_mesh(2,:));
u_exact = u_exact;
% colorbar range
max_u_exact = max(u_exact);
min_u_exact = min(u_exact);


%---------------------%
fig=figure('NumberTitle','off','Name','exact Solution','Renderer', 'painters', 'Position', [0 0 500 600]);
[V_mesh, qx, qy, dx, dy] = generate_mesh(0, 0.5, 0, 1, num_pts);
Ft = TriScatteredInterp(V_mesh(1,:)',V_mesh(2,:)',u_exact');
qz = Ft(qx,qy);

max_u_ite = max(u_exact);
min_u_ite = min(u_exact);

%---------------------%
axes4 = axes(fig);
axes4.Position(1,3) = axes4.Position(1,3)/2;
axes4.Position(1,2) = axes4.Position(1,4)/6*2;
axes4.Position(1,4) = axes4.Position(1,4)/6*4;
xlim([0 0.5]);
xticks([0  0.5]);
yticks([0  1]);
rectangle('position',[0 0 0.5 1] ,'LineWidth', 4);
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',25)
% stringlabel = '$u_1$';
% xlabel({stringlabel},'Interpreter','latex','FontSize',30)
axes2 = axes(fig);
axes2.Position(1,3) = axes4.Position(1,3);
axes2.Position(1,2) = axes4.Position(1,2);
axes2.Position(1,4) = axes4.Position(1,4);

imagesc(axes2,dx,dy,qz);
colormap jet
caxis([min_u_ite max_u_ite])
colorbar off
axis off
h=colorbar('manual');
set(h,'Position',[0.6,0.27,0.025,0.55])
set(h,'Fontsize',25)
set(h,'LineWidth',2)
set(gcf,'color','w')
axis off
savefile = strcat(strcat(strcat(strcat(strcat(strcat('fig',boundary),'-'),algorithm),'-u-exact-'),index),'.png');
saveas(gcf,strcat(savepath,savefile));
% inlarge = axes(fig);
% inlarge.Position = [0.6 0.5 0.15 0.4];
% 
% area = [0.3 0.2 0.5 0.8];
% panpos = inlarge.Position;
% %delete(inlarge);
% [V_mesh, qx, qy, dx, dy] = generate_mesh(0, 1, 0, 1, num_pts);
% u_exact = U(V_mesh(1,:),V_mesh(2,:));
% Ft = TriScatteredInterp(V_mesh(1,:)',V_mesh(2,:)',u_exact');
% qz = Ft(qx,qy);
% inlarge = zoomin(ax,area,panpos);
% title(inlarge,'Zoom in')
% h=colorbar('manual');
% set(h,'Position',[0.8,0.51,0.01,0.38])
% set(h,'Fontsize',25)
% set(h,'LineWidth',2)
% set(gcf,'color','w')
% axis off
%---------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Network predict solution final over entire domain
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% file = strcat(strcat('\u_NN_test_ite',index),'_sub1.mat');
% load(strcat(path,file))
file = strcat(strcat('\u_NN_ite',index),'_test.mat');
load(strcat(path,file))
file = '\u_exact.mat';
load(strcat(path,file))
% u_NN_test = reshape(u_NN_test,[100,100])';
% u_NN_test = reshape(u_NN_test,[10000,1]);
u_NN_test = u_NN_test;
u_NN_left = double(u_NN_test);
% u_NN_left = u_NN_left(end:-1:1);
% mesh
fig=figure('NumberTitle','off','Name','predict Solution','Renderer', 'painters', 'Position', [0 0 500 600]);
[V_mesh, qx, qy, dx, dy] = generate_mesh(0, 0.5, 0, 1, num_pts);
Ft = TriScatteredInterp(V_mesh(1,:)',V_mesh(2,:)',u_NN_left);
qz = Ft(qx,qy);

max_u_ite = max(u_NN_left);
min_u_ite = min(u_NN_left);

%---------------------%
axes4 = axes(fig);
axes4.Position(1,3) = axes4.Position(1,3)/2;
axes4.Position(1,2) = axes4.Position(1,4)/6*2;
axes4.Position(1,4) = axes4.Position(1,4)/6*4;
xlim([0 0.5]);
xticks([0  0.5]);
yticks([0  1]);
rectangle('position',[0 0 0.5 1] ,'LineWidth', 4);
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',25)
% stringlabel = '$\hat{u}_1$';
% xlabel({stringlabel},'Interpreter','latex','FontSize',30)
axes2 = axes(fig);
axes2.Position(1,3) = axes4.Position(1,3);
axes2.Position(1,2) = axes4.Position(1,2);
axes2.Position(1,4) = axes4.Position(1,4);

imagesc(axes2,dx,dy,qz);
colormap jet
caxis([min_u_ite max_u_ite])
colorbar off
axis off
h=colorbar('manual');
set(h,'Position',[0.6,0.27,0.025,0.55])
set(h,'Fontsize',25)
set(h,'LineWidth',2)
set(gcf,'color','w')
axis off
savefile = strcat(strcat(strcat(strcat(strcat(strcat('fig',boundary),'-'),algorithm),'-u-NN-ite-'),index),'.png');
saveas(gcf,strcat(savepath,savefile));

%% pointwise error final over entire domain
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
file = strcat(strcat('\err_test_ite',index),'_sub1.mat');
load(strcat(path,file))
% file = strcat(strcat('\err_test_ite',index),'.mat');
% load(strcat(path,file))
pointerr = abs(double(pointerr1));
min_u_ite = min(pointerr);
max_u_ite = max(pointerr);
% mesh
fig=figure('NumberTitle','off','Name','pointwise error','Renderer', 'painters', 'Position', [0 0 500 600]);
Ft = TriScatteredInterp(V_mesh(1,:)',V_mesh(2,:)',pointerr);
qz = Ft(qx,qy);
axes4 = axes(fig);
axes4.Position(1,3) = axes4.Position(1,3)/2;
axes4.Position(1,2) = axes4.Position(1,4)/6*2;
axes4.Position(1,4) = axes4.Position(1,4)/6*4;
xlim([0 0.5]);
xticks([0  0.5]);
yticks([0  1]);
rectangle('position',[0 0 0.5 1] ,'LineWidth', 4);
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',25)
% stringlabel = '$\hat{u}_1$';
% xlabel({stringlabel},'Interpreter','latex','FontSize',30)
axes2 = axes(fig);
axes2.Position(1,3) = axes4.Position(1,3);
axes2.Position(1,2) = axes4.Position(1,2);
axes2.Position(1,4) = axes4.Position(1,4);

imagesc(axes2,dx,dy,qz);
colormap jet
caxis([min_u_ite max_u_ite])
colorbar off
axis off
h=colorbar('manual');
set(h,'Position',[0.6,0.27,0.025,0.55])
set(h,'Fontsize',25)
set(h,'LineWidth',2)
set(gcf,'color','w')
axis off

savefile = strcat(strcat(strcat(strcat(strcat(strcat('fig',boundary),'-'),algorithm),'-pterr-ite-'),index),'.png');
saveas(gcf,strcat(savepath,savefile));

%% relative pointwise error final over entire domain
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[V_mesh, qx, qy, dx, dy] = generate_mesh(0, 0.5, 0, 1, num_pts);
u_exact_rel = U(V_mesh(1,:),V_mesh(2,:))'+1;
u_NN_test_rel = u_NN_test+1;
relative_pointerr = abs(u_NN_test_rel-u_exact_rel)./u_exact_rel;
u_NN_left = abs(double(relative_pointerr));

% mesh
fig=figure('NumberTitle','off','Name','relative pointwise error','Renderer', 'painters', 'Position', [0 0 500 600]);

[V_mesh, qx, qy, dx, dy] = generate_mesh(0, 0.5, 0, 1, num_pts);
Ft = TriScatteredInterp(V_mesh(1,:)',V_mesh(2,:)',u_NN_left);
qz = Ft(qx,qy);

max_u_ite = max(u_NN_left);
min_u_ite = min(u_NN_left);

%---------------------%
axes4 = axes(fig);
axes4.Position(1,3) = axes4.Position(1,3)/2;
axes4.Position(1,2) = axes4.Position(1,4)/6*2;
axes4.Position(1,4) = axes4.Position(1,4)/6*4;
xlim([0 0.5]);
ylim([0 1]);
xticks([0 0.5]);
yticks([0 1]);
rectangle('position',[0 0 0.5 1] ,'LineWidth', 4);
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',25)
% stringlabel = '$|\frac{u_1-\hat{u}_1}{u_1}|$';
% xlabel({stringlabel},'Interpreter','latex','FontSize',30)
axes2 = axes(fig);
axes2.Position(1,3) = axes4.Position(1,3);
axes2.Position(1,2) = axes4.Position(1,2);
axes2.Position(1,4) = axes4.Position(1,4);

imagesc(axes2,dx,dy,qz);
colormap jet
caxis([min_u_ite max_u_ite])
colorbar off
axis off
h=colorbar('manual');
set(h,'Position',[0.6,0.27,0.025,0.55])
set(h,'Fontsize',25)
set(h,'LineWidth',2)
set(gcf,'color','w')
axis off
savefile = strcat(strcat(strcat(strcat(strcat(strcat('fig',boundary),'-'),algorithm),'-rel-pterr-ite-'),index),'.png');
saveas(gcf,strcat(savepath,savefile));

%% Exact gradient x on final over entire domain
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% file = '\gradu_exact_sub1.mat';
file = '\gradu_exact.mat';
load(strcat(path,file))
% grad_u_exact = gradu_Exact1;
grad_u_exact = grad_u_exact;
u_NN_left = double(grad_u_exact(:,1));

% mesh
fig=figure('NumberTitle','off','Name','Exact grad_x','Renderer', 'painters', 'Position', [0 0 500 600]);
[V_mesh, qx, qy, dx, dy] = generate_mesh(0, 0.5, 0, 1, num_pts);
Ft = TriScatteredInterp(V_mesh(1,:)',V_mesh(2,:)',u_NN_left);
qz = Ft(qx,qy);

max_u_ite = max(u_NN_left);
min_u_ite = min(u_NN_left);

%---------------------%
axes4 = axes(fig);
axes4.Position(1,3) = axes4.Position(1,3)/2;
axes4.Position(1,2) = axes4.Position(1,4)/6*2;
axes4.Position(1,4) = axes4.Position(1,4)/6*4;
xlim([0 0.5]);
xticks([0  0.5]);
yticks([0  1]);
rectangle('position',[0 0 0.5 1] ,'LineWidth', 4);
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',25)
% stringlabel = '$\partial_x\hat{u}_1$';
% xlabel({stringlabel},'Interpreter','latex','FontSize',30)
axes2 = axes(fig);
axes2.Position(1,3) = axes4.Position(1,3);
axes2.Position(1,2) = axes4.Position(1,2);
axes2.Position(1,4) = axes4.Position(1,4);

imagesc(axes2,dx,dy,qz);
colormap jet
caxis([min_u_ite max_u_ite])
colorbar off
axis off
h=colorbar('manual');
set(h,'Position',[0.6,0.27,0.025,0.55])
set(h,'Fontsize',25)
set(h,'LineWidth',2)
set(gcf,'color','w')
axis off

savefile = strcat(strcat(strcat(strcat(strcat(strcat('fig',boundary),'-'),algorithm),'-exact-dx-'),index),'.png');

saveas(gcf,strcat(savepath,savefile));

%% Network predict gradient x on final over entire domain
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% file = strcat(strcat('\gradu_NN_ite',index),'_test.mat');
file = strcat(strcat('\gradu_NN_test_ite',index),'_sub1.mat');
load(strcat(path,file))
% grad_u_test1 = grad_u_test1(10000:-1:1,:);
% grad_u_test1 = grad_u_NN_test1;
u_NN_left = double(grad_u_test1(:,1));

% mesh
fig=figure('NumberTitle','off','Name','predict Solution','Renderer', 'painters', 'Position', [0 0 500 600]);
[V_mesh, qx, qy, dx, dy] = generate_mesh(0, 0.5, 0, 1, num_pts);
Ft = TriScatteredInterp(V_mesh(1,:)',V_mesh(2,:)',u_NN_left);
qz = Ft(qx,qy);

max_u_ite = max(u_NN_left);
min_u_ite = min(u_NN_left);

%---------------------%
axes4 = axes(fig);
axes4.Position(1,3) = axes4.Position(1,3)/2;
axes4.Position(1,2) = axes4.Position(1,4)/6*2;
axes4.Position(1,4) = axes4.Position(1,4)/6*4;
xlim([0 0.5]);
xticks([0  0.5]);
yticks([0  1]);
rectangle('position',[0 0 0.5 1] ,'LineWidth', 4);
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',25)
% stringlabel = '$\partial_x\hat{u}_1$';
% xlabel({stringlabel},'Interpreter','latex','FontSize',30)
axes2 = axes(fig);
axes2.Position(1,3) = axes4.Position(1,3);
axes2.Position(1,2) = axes4.Position(1,2);
axes2.Position(1,4) = axes4.Position(1,4);

imagesc(axes2,dx,dy,qz);
colormap jet
caxis([min_u_ite max_u_ite])
colorbar off
axis off
h=colorbar('manual');
set(h,'Position',[0.6,0.27,0.025,0.55])
set(h,'Fontsize',25)
set(h,'LineWidth',2)
set(gcf,'color','w')
axis off

savefile = strcat(strcat(strcat(strcat(strcat(strcat('fig',boundary),'-'),algorithm),'-dx-ite-'),index),'.png');

saveas(gcf,strcat(savepath,savefile));

%% pointwise error gradient x final over entire domain
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pointerr = grad_u_exact(:,1)-grad_u_test1(:,1);
u_NN_left = double(abs(pointerr));

% mesh
fig=figure('NumberTitle','off','Name','pointwise error','Renderer', 'painters', 'Position', [0 0 500 600]);

Ft = TriScatteredInterp(V_mesh(1,:)',V_mesh(2,:)',u_NN_left);
qz = Ft(qx,qy);

max_u_ite = max(u_NN_left);
min_u_ite = min(u_NN_left);

%---------------------%
axes4 = axes(fig);
axes4.Position(1,3) = axes4.Position(1,3)/2;
axes4.Position(1,2) = axes4.Position(1,4)/6*2;
axes4.Position(1,4) = axes4.Position(1,4)/6*4;
xlim([0 0.5]);
xticks([0  0.5]);
yticks([0  1]);
rectangle('position',[0 0 0.5 1] ,'LineWidth', 4);
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',25)
% stringlabel = '$|\partial_x\hat{u}_1-\partial_x u_1|$';
% xlabel({stringlabel},'Interpreter','latex','FontSize',30)
axes2 = axes(fig);
axes2.Position(1,3) = axes4.Position(1,3);
axes2.Position(1,2) = axes4.Position(1,2);
axes2.Position(1,4) = axes4.Position(1,4);

imagesc(axes2,dx,dy,qz);
colormap jet
caxis([min_u_ite max_u_ite])
colorbar off
axis off
h=colorbar('manual');
set(h,'Position',[0.6,0.27,0.025,0.55])
set(h,'Fontsize',25)
set(h,'LineWidth',2)
set(gcf,'color','w')
axis off
savefile = strcat(strcat(strcat(strcat(strcat(strcat('fig',boundary),'-'),algorithm),'-pterr-dx-ite-'),index),'.png');

saveas(gcf,strcat(savepath,savefile));
%---------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% relative pointwise grad_u_x error final over entire domain
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% file = '\gradu_exact_rel.mat';
% load(strcat(path,file))

iszero = grad_u_exact(:,1) == 0;
grad_u_exact(iszero)=1;

relative_grad_x_pointerr = (grad_u_exact(:,1) - grad_u_test1(:,1))./grad_u_exact(:,1);

u_NN_left = abs(double(relative_grad_x_pointerr));

% mesh
fig=figure('NumberTitle','off','Name','relative grad_u_x pointwise error','Renderer', 'painters', 'Position', [0 0 500 600]);
[V_mesh, qx, qy, dx, dy] = generate_mesh(0, 0.5, 0, 1, num_pts);

Ft = TriScatteredInterp(V_mesh(1,:)',V_mesh(2,:)',u_NN_left);
qz = Ft(qx,qy);

max_u_ite = max(u_NN_left);
min_u_ite = min(u_NN_left);

%---------------------%
axes4 = axes(fig);
axes4.Position(1,3) = axes4.Position(1,3)/2;
axes4.Position(1,2) = axes4.Position(1,4)/6*2;
axes4.Position(1,4) = axes4.Position(1,4)/6*4;
xlim([0 0.5]);
ylim([0 1]);
xticks([0 0.5]);
yticks([0 1]);
rectangle('position',[0 0 0.5 1] ,'LineWidth', 4);
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',25)
% stringlabel = '$|\frac{\partial_x u_1-\partial_x \hat{u}_1}{\partial_x u_1}|$';
% xlabel({stringlabel},'Interpreter','latex','FontSize',30)
axes2 = axes(fig);
axes2.Position(1,3) = axes4.Position(1,3);
axes2.Position(1,2) = axes4.Position(1,2);
axes2.Position(1,4) = axes4.Position(1,4);

imagesc(axes2,dx,dy,qz);
colormap jet
caxis([min_u_ite max_u_ite])
colorbar off
axis off
h=colorbar('manual');
set(h,'Position',[0.6,0.27,0.025,0.55])
set(h,'Fontsize',25)
set(h,'LineWidth',2)
set(gcf,'color','w')
axis off
savefile = strcat(strcat(strcat(strcat(strcat(strcat('fig',boundary),'-'),algorithm),'-rel-pterr-dx-ite-'),index),'.png');
saveas(gcf,strcat(savepath,savefile));

%% Network predict gradient y on final over entire domain
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

u_NN_left = double(grad_u_test1(:,2));

% mesh
fig=figure('NumberTitle','off','Name','predict gradient_y','Renderer', 'painters', 'Position', [0 0 500 600]);
[V_mesh, qx, qy, dx, dy] = generate_mesh(0, 0.5, 0, 1, num_pts);
Ft = TriScatteredInterp(V_mesh(1,:)',V_mesh(2,:)',u_NN_left);
qz = Ft(qx,qy);

max_u_ite = max(u_NN_left);
min_u_ite = min(u_NN_left);

%---------------------%
axes4 = axes(fig);
axes4.Position(1,3) = axes4.Position(1,3)/2;
axes4.Position(1,2) = axes4.Position(1,4)/6*2;
axes4.Position(1,4) = axes4.Position(1,4)/6*4;
xlim([0 0.5]);
xticks([0  0.5]);
yticks([0  1]);
rectangle('position',[0 0 0.5 1] ,'LineWidth', 4);
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',25)
% stringlabel = '$\partial_y\hat{u}_1$';
% xlabel({stringlabel},'Interpreter','latex','FontSize',30)
axes2 = axes(fig);
axes2.Position(1,3) = axes4.Position(1,3);
axes2.Position(1,2) = axes4.Position(1,2);
axes2.Position(1,4) = axes4.Position(1,4);

imagesc(axes2,dx,dy,qz);
colormap jet
caxis([min_u_ite max_u_ite])
colorbar off
axis off
h=colorbar('manual');
set(h,'Position',[0.6,0.27,0.025,0.55])
set(h,'Fontsize',25)
set(h,'LineWidth',2)
set(gcf,'color','w')
axis off

savefile = strcat(strcat(strcat(strcat(strcat(strcat('fig',boundary),'-'),algorithm),'-dy-ite-'),index),'.png');

saveas(gcf,strcat(savepath,savefile));

%% pointwise error gradient y final over entire domain
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pointerr1 = grad_u_test1(:,2)-grad_u_exact(:,2);
u_NN_left = double(abs(pointerr1));

% mesh
fig=figure('NumberTitle','off','Name','pointwise error gradient y','Renderer', 'painters', 'Position', [0 0 500 600]);

Ft = TriScatteredInterp(V_mesh(1,:)',V_mesh(2,:)',u_NN_left);
qz = Ft(qx,qy);

max_u_ite = max(u_NN_left);
min_u_ite = min(u_NN_left);

%---------------------%
axes4 = axes(fig);
axes4.Position(1,3) = axes4.Position(1,3)/2;
axes4.Position(1,2) = axes4.Position(1,4)/6*2;
axes4.Position(1,4) = axes4.Position(1,4)/6*4;
xlim([0 0.5]);
xticks([0  0.5]);
yticks([0  1]);
rectangle('position',[0 0 0.5 1] ,'LineWidth', 4);
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',25)
% stringlabel = '$|\partial_y\hat{u}_1-\partial_y u_1|$';
% xlabel({stringlabel},'Interpreter','latex','FontSize',30)
axes2 = axes(fig);
axes2.Position(1,3) = axes4.Position(1,3);
axes2.Position(1,2) = axes4.Position(1,2);
axes2.Position(1,4) = axes4.Position(1,4);

imagesc(axes2,dx,dy,qz);
colormap jet
caxis([min_u_ite max_u_ite+eps])
colorbar off
axis off
h=colorbar('manual');
set(h,'Position',[0.6,0.27,0.025,0.55])
set(h,'Fontsize',25)
set(h,'LineWidth',2)
set(gcf,'color','w')
axis off
savefile = strcat(strcat(strcat(strcat(strcat(strcat('fig',boundary),'-'),algorithm),'-pterr-dy-ite-'),index),'.png');

saveas(gcf,strcat(savepath,savefile));

%% relative pointwise grad_u_y error final over entire domain
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
relative_grad_y_pointerr = (grad_u_exact_rel(:,2) - grad_u_NN_test_rel(:,2))./grad_u_exact_rel(:,2);
u_NN_left = abs(double(relative_grad_y_pointerr));

% mesh
fig=figure('NumberTitle','off','Name','relative grad_u_y pointwise error','Renderer', 'painters', 'Position', [0 0 500 600]);

[V_mesh, qx, qy, dx, dy] = generate_mesh(0.25, 0.49, 0.2, 0.8, num_pts);
Ft = TriScatteredInterp(V_mesh(1,:)',V_mesh(2,:)',u_NN_left);
qz = Ft(qx,qy);

max_u_ite = max(u_NN_left);
min_u_ite = min(u_NN_left);

%---------------------%
axes4 = axes(fig);
axes4.Position(1,3) = axes4.Position(1,3)/2;
axes4.Position(1,2) = axes4.Position(1,4)/6*2;
axes4.Position(1,4) = axes4.Position(1,4)/6*4;
xlim([0.3 0.5]);
ylim([0.2 0.8]);
xticks([0.3  0.5]);
yticks([0.2  0.8]);
rectangle('position',[0.3 0.2 0.2 0.6] ,'LineWidth', 4);
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',25)
% stringlabel = '$|\frac{\partial_y u_1-\partial_y \hat{u}_1}{\partial_y u_1}|$';
% xlabel({stringlabel},'Interpreter','latex','FontSize',30)
axes2 = axes(fig);
axes2.Position(1,3) = axes4.Position(1,3);
axes2.Position(1,2) = axes4.Position(1,2);
axes2.Position(1,4) = axes4.Position(1,4);

imagesc(axes2,dx,dy,qz);
colormap jet
caxis([min_u_ite max_u_ite])
colorbar off
axis off
h=colorbar('manual');
set(h,'Position',[0.6,0.27,0.025,0.55])
set(h,'Fontsize',25)
set(h,'LineWidth',2)
set(gcf,'color','w')
axis off
savefile = strcat(strcat(strcat(strcat(strcat(strcat('fig',boundary),'-'),algorithm),'-rel-pterr-dy-ite-'),index),'.png');
saveas(gcf,strcat(savepath,savefile));
%---------------------%
function [V_mesh, qx, qy, dx, dy] = generate_mesh(left, right, bottom, top, num_pts)

format short e

% original mesh
mesh_x = linspace(left, right, num_pts);
mesh_y = linspace(bottom, top, num_pts);
% mesh_x = linspace(right, left, num_pts);
% mesh_y = linspace(top, bottom, num_pts);
[mesh_x,mesh_y] = meshgrid(mesh_x,mesh_y);

V_mesh(1,:) = reshape(mesh_x,1,[]);
V_mesh(2,:) = reshape(mesh_y,1,[]);

% interpolate mesh
dx = left:0.002:right;
dy = bottom:0.002:top;
[qx,qy] = meshgrid(dx,dy);

end

function pan = zoomin(ax,areaToMagnify,panPosition,dx,dy,qz)
% AX is a handle to the axes to magnify
% AREATOMAGNIFY is the area to magnify, given by a 4-element vector that defines the
%      lower-left and upper-right corners of a rectangle [x1 y1 x2 y2]
% PANPOSTION is the position of the magnifying pan in the figure, defined by
%        the normalized units of the figure [x y w h]
%

fig = ax.Parent;
pan = copyobj(ax,fig);
if nargin == 6
    axis off
    axes4 = axes(fig);
    imagesc(axes4,dx,dy,qz);
    axis off
    axes4.Position = panPosition;
end
pan.Position = panPosition;
pan.XLim = areaToMagnify([1 3]);
pan.YLim = areaToMagnify([2 4]);
pan.XTick = [];
pan.YTick = [];
rectangle(ax,'Position',...
    [areaToMagnify(1:2) areaToMagnify(3:4)-areaToMagnify(1:2)])
xy = ax2annot(ax,areaToMagnify([1 4;3 2]));
annotation(fig,'line',[xy(1,1) panPosition(1)],...
    [xy(1,2) panPosition(2)+panPosition(4)],'Color','k')
annotation(fig,'line',[xy(2,1) panPosition(1)+panPosition(3)],...
    [xy(2,2) panPosition(2)],'Color','k')
end

function anxy = ax2annot(ax,xy)
% This function converts the axis unites to the figure normalized unites
% AX is a handle to the figure
% XY is a n-by-2 matrix, where the first column is the x values and the
% second is the y values
% ANXY is a matrix in the same size of XY, but with all the values
% converted to normalized units

pos = ax.Position;
%   white area * ((value - axis min) / axis length)   + gray area
normx = pos(3)*((xy(:,1)-ax.XLim(1))./(ax.XLim(2)-ax.XLim(1)))+ pos(1);
normy = pos(4)*((xy(:,2)-ax.YLim(1))./(ax.YLim(2)-ax.YLim(1)))+ pos(2);
anxy = [normx normy];
end