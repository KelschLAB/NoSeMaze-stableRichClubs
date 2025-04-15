% Define weighted adjacency matrix
    A = [0,   0.8,    0,    0;
         0.2,   0,    0.5,    0.5;
         0,   0,   0, 0.5;
         0.1,   0,    0,    0.9];

    % Create directed graph
    G = digraph(A);
    G.Nodes.Name = ["A", "B", "C", "D"]';
    % Compute PageRank (with high damping factor)
    pr = centrality(G, 'pagerank', 'MaxIterations', 1000, 'Importance', G.Edges.Weight, 'FollowProbability', 1);
    newNames = strings(height(G.Nodes), 1);  % string vector
    for i = 1:height(G.Nodes)
        newNames(i) = G.Nodes.Name{i} + ": " + string(round(pr(i), 2));
    end
    G.Nodes.Name = newNames;  % Now valid: string vector
    % Node sizes scaled by PageRank
    nodeSizes = (1 + pr) * 40;

    % Normalize PageRank for color mapping
    pr_norm = rescale(pr);

    % Use built-in 'parula' colormap
    cmap = turbo(256);
    pr_color_idx = round(pr_norm * (size(cmap, 1)-1)) + 1;
    nodeColors = cmap(pr_color_idx, :);

    % Plot graph
    figure('Color', 'w');
    labels = string(round(pr, 2));
    h = plot(G, ...
        'Layout', 'force', ...
        'NodeLabel', round(pr, 2), ...
        'MarkerSize', nodeSizes, ...
        'NodeColor', nodeColors, ...
        'EdgeColor', [0.5 0.5 0.5], ...
        'ArrowSize', 12, ...
        'LineWidth', G.Edges.Weight, ...
        'NodeLabel', G.Nodes.Name);

    % Optionally color edges based on weights
    edge_weights = G.Edges.Weight;
    edge_colors_idx = round(rescale(edge_weights) * (size(cmap,1)-1)) + 1;
    edgeColors = cmap(edge_colors_idx, :);
    h.EdgeColor = edgeColors;

    title('PageRank');
