%% Plot time detected in videos
%   last edited by David Wolf, 30.11.2023
%
%% load dataset
    
clear;
times_detected = {};

for gr = 6:10
    for day = 1:15
        try
            T=readtable(['/zi-flstorage/data/Luise/NoSeMaze2023/DLC_output_windows/G',num2str(gr),'/D',num2str(day),'/p0.8_G',num2str(gr),'_D',num2str(day),'_windowed.csv']);
            assert(size(T,2)==30);
            assert(size(T,1)>100);
        catch
            disp(['Omitted Group ',num2str(gr),', Day ',num2str(day),'. Not found in NoSeMaze2023 output.']);
            continue;
        end
        
        % find start time of recorded video
        try
            original_video_folder = ['/zi-flstorage/data/Shared/Multi_DLC/Sarah_Luise_DLC/Group',num2str(gr),'_Videos/WholeVideo_Analysed/D',num2str(day),'/'];
            original_video = dir(fullfile(original_video_folder,'*.csv'));
            assert(numel(original_video)==1 && ~contains(original_video.name,'comb'));
        catch
            disp(['Omitted Group ',num2str(gr),', Day ',num2str(day),'. Not found in Sarah_Luise_DLC.']);
            continue;
        end
        start_hour = original_video.name(21:22);
        start_min = original_video.name(23:24);
        start_sec = original_video.name(25:26);
        start_in_decimals = str2double(start_hour)+str2double(start_min)/60+str2double(start_sec)/3600;

        count = 1;
        for an = 1:3:28

            % find frames where animal was detected
            detected = find(~isnan(table2array(T(:,an))));

            % convert frames to time 
            F = readtable('/zi-flstorage/data/Luise/NoSeMaze2023/Video_framerates.xlsx');
            framerate = F.(['Group',num2str(gr)])(day);
            tmp = start_in_decimals+detected/framerate/3600;
            
            times_detected{gr,day,count} = cat(1, tmp(tmp<24), tmp(tmp>=24)-24);
            count = count+1;
        end
    end
end

save(fullfile('/zi-flstorage/data/Shared/NoSeMaze/000_social/code for figures/data','times_detected.mat'),'times_detected')