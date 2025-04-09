close all; clear all; clc

% Load data.
load 'klocs4analysis/noGRAPPA_7R_klocs.mat'
dk = klocs_per_xy - klocs_tar_xy;

% Find k-space limits.
minx=round(min(klocs_tar_xy(:,1)));
maxx=round(max(klocs_tar_xy(:,1)));
miny=round(min(klocs_tar_xy(:,2)));
maxy=round(max(klocs_tar_xy(:,2)));
plot_range = 3;

% Downsample for LHS plot.
ds_factor = 5;
Nx = numel(unique(klocs_tar_xy(:,1))); Ny = numel(unique(klocs_tar_xy(:,2)));
[aux1,aux2] = ndgrid(1:ds_factor:Nx,1:ds_factor:Ny);
idx = sub2ind([Nx Ny],aux1(:),aux2(:));

figure(1); cm = linspecer(5); nr = 3; nc = 6; subplot(nr, nc, [1:3 7:9 13:15])
quiver(klocs_tar_xy(:,1), klocs_tar_xy(:,2), dk(:,1), dk(:,2), 'Color', [0.5 0.5 0.5], 'LineWidth', 0.5); hold on
quiver(klocs_tar_xy(idx,1), klocs_tar_xy(idx,2), dk(idx,1), dk(idx,2), 'Color', 'k', 'LineWidth', 1);
grid on; grid minor; set(gca,'FontSize',16,'XTick',-50:10:50,'XTickLabels',{'','-40','','-20','','0','','20','','',''},'YTick',-50:10:50,'YTickLabels',{'','-40','','-20','','0','','20','','40',''})
xlabel('k_x index', 'FontSize', 18,'Position',[40,-51,-1]); ylabel('k_y index', 'FontSize', 18)
xlim([minx-plot_range,maxx+plot_range]); ylim([minx-plot_range,maxx+plot_range]); % Enforce symmetric...

title('no GRAPPA','FontSize',22,'FontWeight','bold','Position',[50.7,50.8,0])

%% Draw colored boxes to represent zoomed regions.

ds_factor2 = 4;

rectangle('Position', [minx, maxy-(2*plot_range-1), Nx, 2*plot_range], 'EdgeColor', cm(1,:), 'LineWidth', 2)
rectangle('Position', [minx, -plot_range, Nx, 2*plot_range], 'EdgeColor', cm(2,:), 'LineWidth', 2)
rectangle('Position', [minx, miny, Nx, 2*plot_range], 'EdgeColor', cm(3,:), 'LineWidth', 2)

% Smaller plots with colored axes.
subplot(nr, nc, 4:6)
quiver(klocs_tar_xy(1:ds_factor2:end,1), klocs_tar_xy(1:ds_factor2:end,2), dk(1:ds_factor2:end,1), dk(1:ds_factor2:end,2), 'k', 'LineWidth', 1)
set(gca,'FontSize',12,'YTickLabels',{'42','44','46',''}); grid on; grid minor; xlim([minx maxx]); ylim([maxy-(2*plot_range-1) maxy+1]); coloredAxes(gca, cm(1,:))

subplot(nr, nc, 10:12)
quiver(klocs_tar_xy(1:ds_factor2:end,1), klocs_tar_xy(1:ds_factor2:end,2), dk(1:ds_factor2:end,1), dk(1:ds_factor2:end,2), 'k', 'LineWidth', 1)
set(gca,'FontSize',12); grid on; grid minor; xlim([minx maxx]); ylim([-2*plot_range 2*plot_range]); coloredAxes(gca, cm(2,:))

subplot(nr, nc, 16:18)
quiver(klocs_tar_xy(1:ds_factor2:end,1), klocs_tar_xy(1:ds_factor2:end,2), dk(1:ds_factor2:end,1), dk(1:ds_factor2:end,2), 'k', 'LineWidth', 1)
set(gca,'FontSize',12); grid on; grid minor; xlim([minx maxx]); ylim([miny miny+(2*plot_range)]); coloredAxes(gca, cm(3,:))

set(gcf,'Color','w','Position',[1,49,1280,599.33])

% Function to change the color of axes in smaller plots
function coloredAxes(ax, color)
ax.XColor = color;
ax.YColor = color;
end