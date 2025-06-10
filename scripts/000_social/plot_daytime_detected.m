%% Plot daytime of videodetections as polarhistogram
% last edited by David Wolf, 30.11.2023
%
%% load data

clear;
save_dir = 'W:\group_entwbio\data\Shared\NoSeMaze\000_social\code for figures\figures';

% times_detected is a cell array of dimensions: group x day x animal
load(fullfile('W:\group_entwbio\data\Shared\NoSeMaze\000_social\code for figures\data','times_detected.mat'));

% pool all detections from all animals
collect_daytime = [];
for gr = 1:size(times_detected,1)
    for day = 1:size(times_detected,2)
        for an = 1:size(times_detected,3)
            collect_daytime = cat(1,collect_daytime, times_detected{gr,day,an});
        end
    end
end


%% plot

f = figure;
h = polarhistogram(collect_daytime*(2*pi/24),24,'Normalization','probability',...
    'FaceColor',[227 30 36]./255,'FaceAlpha',1);
h.EdgeColor = 'k'; %[227 30 36]./255;
h.LineWidth = 1;
axis = gca;
axis.ThetaTickLabel = cellfun(@num2str,num2cell((0:2:22)'),'UniformOutput',0);
axis.FontSize = 6;
set(gcf, 'Units', 'centimeters');
set(gcf, 'Position', [3 3 4.5 3]);
exportgraphics(gcf, fullfile(save_dir,'detections_over_daytime.pdf'),'ContentType','vector','BackgroundColor','none');
close all;

% export source data
%writetable(array2table(collect_daytime),fullfile(save_dir,'detections_over_daytime_source.xlsx'));

