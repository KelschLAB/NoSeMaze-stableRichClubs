% simple flow from genotype to RC/non-RC
figure('Name','sankey demo1','Units','normalized','Position',[.05,.2,.5,.56])
links={'WT','RC',29;'Mutant','RC',2;'WT','Non-RC', 78;'Mutant','Non-RC',30};
SK=SSankey(links(:,1),links(:,2),links(:,3));
SK.ColorList=[0.0, 0.7, 0.0;
    0.8, 0.0, 0.0;
    0.0, 0.0, 1.0;
    0.7, 0.7, 0.7;];
SK.RenderingMethod='interp';  
SK.Align='center';
SK.LabelLocation='top';
SK.Sep=.2;
SK.draw()

% simple flow from genotype to RC/non-RC Normalized
figure('Name','sankey demo1','Units','normalized','Position',[.05,.2,.5,.56])
links={'WT','RC', 0.3718;'Mutant','RC', 0.0667;'WT','Non-RC', 0.6282;'Mutant','Non-RC', 0.9333};
SK=SSankey(links(:,1),links(:,2),links(:,3));
SK.ColorList=[0.0, 0.7, 0.0;
    0.8, 0.0, 0.0;
    0.0, 0.0, 1.0;
    0.7, 0.7, 0.7;];
SK.RenderingMethod='interp';  
SK.Align='center';
SK.LabelLocation='top';
SK.Sep=.2;
SK.draw()

% 1 step shuffling flow chart from RC/non-RC/mutants
corr_rc_to_rc = 0.25;
RC =1;
non_RC = 1;

figure('Name','sankey demo1','Units','normalized','Position',[.05,.2,.5,.56])
links={'RC','RC ', corr_rc_to_rc; 'Non-RC','RC ', 1-corr_rc_to_rc; 'Non-RC','Non-RC ', corr_rc_to_rc
    'Mutant','RC ', 0; 'RC','Non-RC ', (1-corr_rc_to_rc)*RC; 'Mutant', 'Non-RC ', 1};

SK=SSankey(links(:,1),links(:,2),links(:,3));
SK.ColorList=[0.0, 0.0, 0.8;
    0.0, 0.8, 0.0;
    0.8, 0.0, 0.0;
    0.0, 0.0, 0.8;
    0.7, 0.7, 0.7;];

SK.RenderingMethod='interp';  
SK.Align='center';
SK.LabelLocation='top';
SK.Sep=.2;
SK.draw()

% 3 step flow from genotype to RC/non-RC with reshuffling
corr_rc_to_rc = 0.25;
WT_to_RC = 0.37;
mutant_to_RC = 0.07;
RC = WT_to_RC + mutant_to_RC;
non_RC = 2 - RC;
mutants_in_total = 0.23;
non_rc_to_rc = (1-corr_rc_to_rc)*RC;

figure('Name','sankey demo1','Units','normalized','Position',[.05,.2,.5,.56])
links={'WT','RC ', WT_to_RC; 'Mutant','RC ', mutant_to_RC ; 'WT','Non-RC ', 1-WT_to_RC; 'Mutant', 'Non-RC ', 1-mutant_to_RC
    'RC ', 'RC', corr_rc_to_rc*RC; 'RC ', 'Non-RC', (1-corr_rc_to_rc)*RC; 
    'Non-RC ', 'RC', non_rc_to_rc; 'Non-RC ', 'Non-RC', (1-(non_rc_to_rc/non_RC))*non_RC
    'RC','WT ',corr_rc_to_rc*RC+ (non_rc_to_rc/non_RC)*non_RC; 'RC', 'Mutant ', 0.0; 'Non-RC', 'WT ', (1-(non_rc_to_rc/non_RC))*non_RC+(1-corr_rc_to_rc)*RC-1; 'Non-RC', 'Mutant ', 1};

SK=SSankey(links(:,1),links(:,2),links(:,3));
SK.ColorList=[0.0, 0.7, 0.0;
    0.8, 0.0, 0.0;
    0.0, 0.0, 1.0;
    0.7, 0.7, 0.7;
    0.0, 0.0, 1.0;
    0.7, 0.7, 0.7;
    0.0, 0.7, 0.0;
    0.8, 0.0, 0.0;];
SK.RenderingMethod='interp';  
SK.Align='center';
SK.LabelLocation='top';
SK.Sep=.2;
SK.draw()
