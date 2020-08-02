% Uses the precomputed data for neighborhoods and flows
%

function alpha = IFMpcd(IFMdata)

    IFMdata.CM_inInd = double(IFMdata.CM_inInd);
    IFMdata.CM_neighInd = double(IFMdata.CM_neighInd);
    IFMdata.CM_flows = double(IFMdata.CM_flows);
    IFMdata.LOC_inInd = double(IFMdata.LOC_inInd);
    IFMdata.LOC_flows1 = double(IFMdata.LOC_flows1);
    IFMdata.LOC_flows2 = double(IFMdata.LOC_flows2);
    IFMdata.LOC_flows3 = double(IFMdata.LOC_flows3);
    IFMdata.LOC_flows4 = double(IFMdata.LOC_flows4);
    IFMdata.LOC_flows5 = double(IFMdata.LOC_flows5);
    IFMdata.LOC_flows6 = double(IFMdata.LOC_flows6);
    IFMdata.LOC_flows7 = double(IFMdata.LOC_flows7);
    IFMdata.LOC_flows8 = double(IFMdata.LOC_flows8);
    IFMdata.LOC_flows9 = double(IFMdata.LOC_flows9);
    IFMdata.IU_inInd = double(IFMdata.IU_inInd);
    IFMdata.IU_neighInd = double(IFMdata.IU_neighInd);
    IFMdata.IU_flows = double(IFMdata.IU_flows);
    IFMdata.kToU = double(IFMdata.kToU);
    IFMdata.kToUconf = double(IFMdata.kToUconf);
    IFMdata.known = double(IFMdata.known);

    [h, w, ~] = size(IFMdata.kToU);
    N = h * w;

    CM_weights = ones(h, w);
    LOC_weights = ones(h, w);
    IU_weights = ones(h, w);
    KU_weights = ones(h, w);
    %CM_weights(:, 1 : ceil(w/2)) = 0;
    %LOC_weights(:, 1 : ceil(w/2)) = 0;
    %KU_weights(:, 1 : ceil(w/2)) = 0;
    %IU_weights(100 : 250, 450 : 600) = 0;
    %

    mattingLaplacian2(N, h, w, IFMdata.LOC_inInd, IFMdata.LOC_flows1, ...
                IFMdata.LOC_flows2, IFMdata.LOC_flows3, IFMdata.LOC_flows4, IFMdata.LOC_flows5, ...
                IFMdata.LOC_flows6, IFMdata.LOC_flows7, IFMdata.LOC_flows8, IFMdata.LOC_flows9, ...
                LOC_weights)

    % cm_mult = 1;
    % loc_mult = 1;
    % iu_mult = 0.01;
    % ku_mult = 0.05;
    % lambda = 100;
    % A = ...
    %             cm_mult * colorMixtureLaplacian(N, IFMdata.CM_inInd, IFMdata.CM_neighInd, IFMdata.CM_flows, CM_weights) + ...
    %             loc_mult * mattingLaplacian2(N, h, IFMdata.LOC_inInd, IFMdata.LOC_flows1, ...
    %             IFMdata.LOC_flows2, IFMdata.LOC_flows3, IFMdata.LOC_flows4, IFMdata.LOC_flows5, ...
    %             IFMdata.LOC_flows6, IFMdata.LOC_flows7, IFMdata.LOC_flows8, IFMdata.LOC_flows9, ...
    %             LOC_weights) + ...
    %             iu_mult * similarityLaplacian(N, IFMdata.IU_inInd, IFMdata.IU_neighInd, IFMdata.IU_flows, IU_weights) + ... 
    %             ku_mult * spdiags(KU_weights(:), 0, N, N) * spdiags(IFMdata.kToUconf(:), 0, N, N) + ...
    %             lambda * spdiags(double(IFMdata.known(:)), 0, N, N);
    %             %loc_mult * mattingLaplacian2(N, h, IFMdata.LOC_inInd, IFMdata.LOC_flows, LOC_weights) + ...
    %             %loc_mult * mattingLaplacian(N, IFMdata.LOC_inInd, IFMdata.LOC_flowRows, IFMdata.LOC_flowCols, IFMdata.LOC_flows, LOC_weights) + ...
    %
    % b =  ( ...
    %             ku_mult * spdiags(KU_weights(:), 0, N, N) * spdiags(IFMdata.kToUconf(:), 0, N, N) + ...
    %             lambda * spdiags(double(IFMdata.known(:)), 0, N, N) ...
    %         ) * IFMdata.kToU(:);
    %
    % alpha = pcg(A, b, [], 2000);
    %
    % alpha(alpha < 0) = 0;
    % alpha(alpha > 1) = 1;
    % alpha = reshape(alpha, [h, w]);
end

function Lcm = colorMixtureLaplacian(N, inInd, neighInd, flows, weights)
    Wcm = sparse(repmat(inInd(:), [1 size(flows, 2)]), neighInd(:), flows, N, N);
    Wcm = spdiags(weights(:), 0 , N, N) * Wcm;
    Lcm = spdiags(sum(Wcm, 2), 0 , N, N) - Wcm;
    Lcm = Lcm' * Lcm;
end

function Lmat = mattingLaplacian2(N, h, w, inInd, flows1, flows2, flows3, flows4, flows5, flows6, flows7, flows8, flows9, weights)
    load('../../indices.mat');
    converted = double(py2matlab(pyLOC_inInd, h, w));
    d = max(abs(converted-inInd));
    assert(d == 0);

    weights = weights(inInd);
    %flows = flows .* repmat(reshape(weights, 1, 1, length(weights)) , [size(flows, 1) size(flows, 1) 1]);
    weights = repmat(reshape(weights, length(weights), 1) , [1 9]);
    flows1 = flows1 .* weights;
    flows2 = flows2 .* weights;
    flows3 = flows3 .* weights;
    flows4 = flows4 .* weights;
    flows5 = flows5 .* weights;
    flows6 = flows6 .* weights;
    flows7 = flows7 .* weights;
    flows8 = flows8 .* weights;
    flows9 = flows9 .* weights;

    
    % Define local neighborhood
    neighInds = [(inInd-h-1) (inInd-h) (inInd-h+1) (inInd-1) (inInd) (inInd+1) (inInd+h-1) (inInd+h) (inInd+h+1)];
    load('../../neigh_indices.mat');
    converted = double(py2matlab(pyLOC_neighInd, h, w));
    d = abs(converted-neighInds);
    assert(max(d(:)) == 0);

    Wmat = sparse(N, N);
    % We'll create a separate sparse matrix for each column of 9x9xlength(inInd) flows tensor

    iRows = repmat(neighInds(:, 1), [1 9]);
    Wmat = Wmat + sparse(iRows(:), neighInds(:), flows1(:), N, N);

    load('../../Wmat1.mat');
    converted = double(py2matlab(py_wmat1_irows, h, w));
    % d = abs(converted-iRows);
    % assert(max(d(:)) == 0);
    row = double(py2matlab(py_wmat1_row, 497, 800));
    col = double(py2matlab(py_wmat1_col, 497, 800));
    Matpy = sparse(row, col , double(py_wmat1_data), N, N);
    dif = Wmat-Matpy;
    assert(max(abs(dif(:))) < 1e-8)

    iRows = repmat(neighInds(:, 2), [1 9]);
    iWmat = sparse(iRows(:), neighInds(:), flows2(:), N, N);
    Wmat = Wmat + iWmat;

    load('../../Wmat2.mat');
    converted = double(py2matlab(py_wmat2_irows, h, w));
    % d = abs(converted-iRows);
    % assert(max(d(:)) == 0);
    row = double(py2matlab(py_wmat2_row, 497, 800));
    col = double(py2matlab(py_wmat2_col, 497, 800));
    Matpy = sparse(row, col , double(py_wmat2_data), N, N);
    dif = iWmat-Matpy;
    max(abs(dif(:)))
    keyboard

    iRows = repmat(neighInds(:, 3), [1 9]);
    Wmat = Wmat + sparse(iRows(:), neighInds(:), flows3(:), N, N);
    iRows = repmat(neighInds(:, 4), [1 9]);
    Wmat = Wmat + sparse(iRows(:), neighInds(:), flows4(:), N, N);
    iRows = repmat(neighInds(:, 5), [1 9]);
    Wmat = Wmat + sparse(iRows(:), neighInds(:), flows5(:), N, N);
    iRows = repmat(neighInds(:, 6), [1 9]);
    Wmat = Wmat + sparse(iRows(:), neighInds(:), flows6(:), N, N);
    iRows = repmat(neighInds(:, 7), [1 9]);
    Wmat = Wmat + sparse(iRows(:), neighInds(:), flows7(:), N, N);
    iRows = repmat(neighInds(:, 8), [1 9]);
    Wmat = Wmat + sparse(iRows(:), neighInds(:), flows8(:), N, N);
    iRows = repmat(neighInds(:, 9), [1 9]);
    Wmat = Wmat + sparse(iRows(:), neighInds(:), flows9(:), N, N);



    % for i = 1 : 9
    %     iRows = repmat(neighInds(:, i), [1 9]);
    %     iFlows = flows(:, i, :);
    %     iFlows = permute(iFlows, [3 1 2]);
    %     Wmat = Wmat + sparse(iRows(:), neighInds(:), iFlows(:), N, N);
    % end

    load('../../dump.mat');
    row = double(py2matlab(py_row, 497, 800));
    col = double(py2matlab(py_col, 497, 800));
    Matpy = sparse(row, col , double(py_data), N, N);
    dif = Wmat-Matpy;
    max(abs(dif(:)))
    
    Wmat = (Wmat + Wmat') / 2; % Make symmetric
    rowsum = sum(Wmat, 2);
    Lmat = spdiags(rowsum, 0 , N, N) - Wmat;

end

function Lmat = mattingLaplacian(N, inInd, flowRows, flowCols, flows, weights)
    weights = weights(inInd);
    flows = flows .* repmat(reshape(weights, 1, 1, length(weights)) , [size(flows, 1) size(flows, 1) 1]);
    Wmat = sparse(flowRows(:), flowCols(:), flows(:), N, N);
    Wmat = (Wmat + Wmat') / 2; % Make symmetric
    Lmat = spdiags(sum(Wmat, 2), 0 , N, N) - Wmat;
end

function Lcs = similarityLaplacian(N, inInd, neighInd, flows, weights)
    weights = weights(inInd);
    flows = flows .* repmat(weights, [1 size(flows, 2)]);
    inInd = repmat(inInd, [1, size(neighInd, 2)]);
    Wcs = sparse(inInd(:), neighInd(:), flows, N, N);
    Wcs = (Wcs + Wcs') / 2; % Make symmetric
    Lcs = spdiags(sum(Wcs, 2), 0 , N, N) - Wcs;
end
