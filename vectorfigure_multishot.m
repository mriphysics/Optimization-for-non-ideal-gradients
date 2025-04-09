close all; clear all; clc

% Load data.
load 'klocs4analysis/multishot_11R_klocs.mat'

cm_shots = linspecer(16);
Nshots = 16;
Nsamples = size(klocs_tar_xy,1)/Nshots;

% Find k-space limits.
minx=round(min(klocs_tar_xy(:,1)));
maxx=round(max(klocs_tar_xy(:,1)));
miny=round(min(klocs_tar_xy(:,2)));
maxy=round(max(klocs_tar_xy(:,2)));
plot_range = 3; sf = 0.5;

ds_factor = 5;

for ii = 1:Nshots

    klocs_opt_xy_plot = klocs_opt_xy((ii-1)*Nsamples+1:ii*Nsamples,:);
    klocs_per_xy_plot = klocs_per_xy((ii-1)*Nsamples+1:ii*Nsamples,:);
    klocs_tar_xy_plot = klocs_tar_xy((ii-1)*Nsamples+1:ii*Nsamples,:);

    dk = klocs_per_xy_plot - klocs_tar_xy_plot;

    figure(1); nr = 3; nc = 6; subplot(nr, nc, [1:3 7:9 13:15])
    quiver(klocs_tar_xy_plot(1:ds_factor:end,1), klocs_tar_xy_plot(1:ds_factor:end,2), dk(1:ds_factor:end,1), dk(1:ds_factor:end,2), sf, 'Color', cm_shots(ii,:), 'LineWidth', 1); hold on
    grid on; grid minor; set(gca,'FontSize',16,'XTick',-60:10:60,'XTickLabels',{'','','-40','','-20','','0','','20','','','',''},'YTick',-60:10:60,'YTickLabels',{'-60','','-40','','-20','','0','','20','','40','','60'})
    xlabel('k_x index', 'FontSize', 18,'Position',[50,-67,-1]); ylabel('k_y index', 'FontSize', 18)
    xlim([minx-plot_range,maxx+plot_range]); ylim([minx-plot_range,maxx+plot_range]); % Enforce symmetric...

end

title('Multishot','FontSize',22,'FontWeight','bold','Position',[68,67,0])

%% Draw colored boxes to represent zoomed regions.

Nx = numel(unique(klocs_tar_xy(:,1))); Ny = numel(unique(klocs_tar_xy(:,2)));

ds_factor2 = 4;

rectangle('Position', [minx, maxy-(2*plot_range-1), Nx, 2*plot_range], 'EdgeColor', 'k', 'LineWidth', 2)
rectangle('Position', [minx, -plot_range, Nx, 2*plot_range], 'EdgeColor', 'k', 'LineWidth', 2)
rectangle('Position', [minx, miny, Nx, 2*plot_range], 'EdgeColor', 'k', 'LineWidth', 2)

for ii = 1:Nshots

    klocs_opt_xy_plot = klocs_opt_xy((ii-1)*Nsamples+1:ii*Nsamples,:);
    klocs_per_xy_plot = klocs_per_xy((ii-1)*Nsamples+1:ii*Nsamples,:);
    klocs_tar_xy_plot = klocs_tar_xy((ii-1)*Nsamples+1:ii*Nsamples,:);

    dk = klocs_per_xy_plot - klocs_tar_xy_plot;

    subplot(nr, nc, 4:6)
    quiver(klocs_tar_xy_plot(1:ds_factor:end,1), klocs_tar_xy_plot(1:ds_factor:end,2), dk(1:ds_factor:end,1), dk(1:ds_factor:end,2), sf, 'Color', cm_shots(ii,:), 'LineWidth', 1); hold on
    grid on; grid minor; xlim([minx maxx]); ylim([maxy-(2*plot_range-1) maxy+1]); set(gca,'FontSize',12,'YTickLabels',{'58','60','62',''}); 

end

for ii = 1:Nshots

    klocs_opt_xy_plot = klocs_opt_xy((ii-1)*Nsamples+1:ii*Nsamples,:);
    klocs_per_xy_plot = klocs_per_xy((ii-1)*Nsamples+1:ii*Nsamples,:);
    klocs_tar_xy_plot = klocs_tar_xy((ii-1)*Nsamples+1:ii*Nsamples,:);

    dk = klocs_per_xy_plot - klocs_tar_xy_plot;

    subplot(nr, nc, 10:12)
    quiver(klocs_tar_xy_plot(1:ds_factor:end,1), klocs_tar_xy_plot(1:ds_factor:end,2), dk(1:ds_factor:end,1), dk(1:ds_factor:end,2), sf, 'Color', cm_shots(ii,:), 'LineWidth', 1); hold on
    grid on; grid minor; xlim([minx maxx]); ylim([-2*plot_range 2*plot_range]); set(gca,'FontSize',12); 

end

for ii = 1:Nshots

    klocs_opt_xy_plot = klocs_opt_xy((ii-1)*Nsamples+1:ii*Nsamples,:);
    klocs_per_xy_plot = klocs_per_xy((ii-1)*Nsamples+1:ii*Nsamples,:);
    klocs_tar_xy_plot = klocs_tar_xy((ii-1)*Nsamples+1:ii*Nsamples,:);

    dk = klocs_per_xy_plot - klocs_tar_xy_plot;

    subplot(nr, nc, 16:18)
    quiver(klocs_tar_xy_plot(1:ds_factor:end,1), klocs_tar_xy_plot(1:ds_factor:end,2), dk(1:ds_factor:end,1), dk(1:ds_factor:end,2), sf, 'Color', cm_shots(ii,:), 'LineWidth', 1); hold on
    grid on; grid minor; xlim([minx maxx]); ylim([miny miny+(2*plot_range)]); set(gca,'FontSize',12); 

end

set(gcf,'Color','w','Position',[1,49,1280,599.33])
