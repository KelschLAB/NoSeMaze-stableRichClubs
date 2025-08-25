%% approach and number of interactions emerged as candidates for mutant effects -> visualize networks to understand potential effect
save_plots = '/zi-flstorage/data/Shared/NoSeMaze/2_mutant/plots/sn_analysis/';
parent_path = '/zi-flstorage/data/Luise/NoSeMaze2023/DLC_output_windows/';


networks_of_interest = {'approaches_resD1_','interactions_resD1_'};

for nn = 1:numel(networks_of_interest)
    for gr = 1:17
        
        table_path = fullfile(parent_path,['G',num2str(gr)],'adjacency_matrix');

        f= figure;
        for dd = 1:14
            subplot(2,7,dd);
            cur_path = fullfile(table_path,[networks_of_interest{nn},num2str(dd),'.csv']);
            if isfile(cur_path)
                input_table = readtable(cur_path);
                input_table = sortrows(input_table);
                input_table = input_table(:,sort(input_table.Properties.VariableNames));
                adjacency_matrix = table2array(input_table(:,2:end));
                names = input_table.Properties.VariableNames(2:end);
                names = cellfun(@(x) x(2:end), names,'UniformOutput',0);
                
                % if last row&col all zeros resort, because of matlab
                % issues otherwise
                sumcheck = sum(adjacency_matrix,1)+sum(adjacency_matrix,2);
                while sumcheck(end)==0
                    resort_idx = randperm(size(adjacency_matrix,1));
                    tmp = adjacency_matrix(:,resort_idx);
                    tmp2 = tmp(resort_idx,:);
                    adjacency_matrix = tmp2;
                    names = names(resort_idx);
                    sumcheck = sum(adjacency_matrix,1)+sum(adjacency_matrix,2);

                end
                
                
                % find mutant status
                group = zeros(1,numel(names));
                for ii = 1:numel(names)
                    cur_status = full_data.mutant(contains(full_data.Mouse_RFID,names{ii}) & full_data.group==gr);
                    if ~isempty(cur_status) && cur_status
                        group(ii) = 1;
                    end
                end
                
                p = plot_network_from_adjacency_matrix(adjacency_matrix,names,group);
                title(['Day ',num2str(dd)]);
            end
        end
        
        sgtitle({networks_of_interest{nn}(1:end-7),['Group ',num2str(gr)]});
        
        f.Units = 'centimeters';
        f.Position = [0 0 67 35];
        exportgraphics(f,fullfile(save_plots,[networks_of_interest{nn},'_group_',num2str(gr),'.png']));
        exportgraphics(f,fullfile(save_plots,[networks_of_interest{nn},'_group_',num2str(gr),'.pdf']));
        close all;
    end
end
    


%% understand centrality measures visually

exclude = cellfun(@numel,full_data.eigenvector_int_count_evolution)<14;
ctr = cell2mat(full_data.eigenvector_int_count_evolution(~full_data.mutant & ~exclude));
mt = cell2mat(full_data.eigenvector_int_count_evolution(full_data.mutant & ~exclude));
ctr(isinf(ctr)) = NaN; mt(isinf(mt)) = NaN;

% who has the highest score?
[M,I] = max(ctr(:,1));
ctr_idx = find(~full_data.mutant & ~exclude);
ctr_idx(I)

%% correlate first day hierarchy to later social integration

window_size = 1; % days

for gr = 1:21

    % import hierarchy from integrated Tube-Test (ITT)
    load(fullfile(data_dir, ['full_hierarchy_G',num2str(gr),'.mat']));
    DS_evolution = [];
    
    % moving average over x days for computation of hierarchy
    for dd = 1:min([14,numel(full_hierarchy)-window_size])
        
        cur_match_matrix = zeros(size(full_hierarchy(1).match_matrix));
        for ii = 1:window_size
            cur_match_matrix = cur_match_matrix + full_hierarchy((dd-1)+ii).match_matrix;
        end
        
        current_DS = compute_DS_from_match_matrix(cur_match_matrix);
        DS_evolution = cat(2,DS_evolution,current_DS.DS');
    end
    
    % parse to animals
    for an = 1:size(DS_evolution,1)
       
        animal_idx = find(contains(full_data.Mouse_RFID,full_hierarchy(1).ID{an}) & full_data.group==gr);
        if ~isempty(animal_idx)
            full_data.(['hierarchy_evolution_',num2str(window_size)]){animal_idx} = DS_evolution(an,:);
        end
    end

end

save_plots = '/zi-flstorage/data/Shared/NoSeMaze/2_mutant/plots/correlate_sna_to_stuff/';
variable_names = full_data.Properties.VariableNames;
features_of_interest = find(contains(variable_names,{'eigenvector_int_count','KC_hub_abs_approach'}));

for ff = features_of_interest
    exclude = cellfun(@numel,full_data.(variable_names{ff}))<14;
    foi = cellfun(@(x) mean(x(5:10),'omitnan'),table2cell(full_data(~exclude,ff)));


    %%% correlate to first day hierarchy
    foc = cellfun(@(x) x(1),full_data.hierarchy_evolution_1(~exclude));

    f = figure;
    hold on;
    scatter(foi(~full_data.mutant(~exclude)),foc(~full_data.mutant(~exclude)));
    scatter(foi(full_data.mutant(~exclude)),foc(full_data.mutant(~exclude)));
    legend('ctr','mt');
    xlabel(variable_names{ff},'Interpreter','none');
    ylabel('DS_tube day 1','Interpreter','none');
    l=lsline;
    l(2).Color = 'b'; l(1).Color = 'r';
    [rho,pval] = corr(foi(~full_data.mutant(~exclude)),foc(~full_data.mutant(~exclude)),'rows','complete');
    [rho_mt,pval_mt] = corr(foi(full_data.mutant(~exclude)),foc(full_data.mutant(~exclude)),'rows','complete');
    title({['rho = ',num2str(round(rho,3)),'. p = ',num2str(round(pval,3)),'. n = ',num2str(numel(foc(~full_data.mutant(~exclude))))],...
        ['rho (mt) = ',num2str(round(rho_mt,3)),'. p = ',num2str(round(pval_mt,3)),'. n = ',num2str(numel(foc(full_data.mutant(~exclude))))]});
    exportgraphics(f,fullfile(save_plots,[variable_names{ff},'_vs_DStube.png']));
    close all;

    f = figure;
    hold on;
    scatter(foi,foc);
    xlabel(variable_names{ff},'Interpreter','none');
    ylabel('DS_tube day 1','Interpreter','none');
    l=lsline;
    [rho,pval] = corr(foi,foc,'rows','complete');
    title(['rho = ',num2str(round(rho,3)),'. p = ',num2str(round(pval,3)),'. n = ',num2str(numel(foc))]);
    exportgraphics(f,fullfile(save_plots,[variable_names{ff},'_vs_DStube_global.png']));
    close all;


end
    




%% correlate to stuff
save_plots = '/zi-flstorage/data/Shared/NoSeMaze/2_mutant/plots/correlate_sna_to_stuff/';
variable_names = full_data.Properties.VariableNames;
features_of_interest = find(contains(variable_names,{'eigenvector_int_count','KC_hub_abs_approach'}));



for ff = features_of_interest 
      
    exclude = cellfun(@numel,full_data.(variable_names{ff}))<14;
    foi = cellfun(@(x) mean(x(5:10),'omitnan'),table2cell(full_data(~exclude,ff)));

    f = figure;
    hold on;
    histogram(foi(~full_data.mutant(~exclude)),'BinWidth',0.2,'Normalization','probability');
    histogram(foi(full_data.mutant(~exclude)),'BinWidth',0.2,'Normalization','probability');
    exportgraphics(f,fullfile(save_plots,[variable_names{ff},'.png']));
    close all;
    
    %%% correlate to aggregated factors
    for ii = 11:45
               
        foc = table2array(full_data(~exclude,ii));
        
        f = figure;
        hold on;
        scatter(foi(~full_data.mutant(~exclude)),foc(~full_data.mutant(~exclude)));
        scatter(foi(full_data.mutant(~exclude)),foc(full_data.mutant(~exclude)));
        legend('ctr','mt');
        xlabel(variable_names{ff},'Interpreter','none');
        ylabel(variable_names{ii},'Interpreter','none');
        lsline
        [rho,pval] = corr(foi,foc,'rows','complete');
        title({['rho = ',num2str(round(rho,3))],['p = ',num2str(round(pval,3))]});
        exportgraphics(f,fullfile(save_plots,[variable_names{ff},'_vs_',variable_names{ii},'.png']));
        close all;
        
        
    end
    
end



