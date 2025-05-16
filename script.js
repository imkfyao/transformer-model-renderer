// --- Global ID counter for Vis.js nodes ---
let visNodeIdCounter = 1;
function getNextVisNodeId() {
    return `vis_node_${visNodeIdCounter++}`;
}
let visEdgeIdCounter = 1;
function getNextVisEdgeId() {
    return `vis_edge_${visEdgeIdCounter++}`;
}

function parseLayerConfigString(layerStr) {
    const config = { type: '', numExperts: null };
    const s = layerStr.trim().toLowerCase();
    if (s.includes(':')) {
        const parts = s.split(':');
        config.type = parts[0];
        if (config.type === 'moe') {
            config.numExperts = parseInt(parts[1], 10);
            if (isNaN(config.numExperts) || config.numExperts <= 0) {
                console.warn(`Invalid expert count for MoE: ${s}. Defaulting to 8.`);
                config.numExperts = 8;
            }
        } else {
            console.warn(`Suffix found for non-MoE layer '${s}'. Ignoring suffix.`);
        }
    } else {
        config.type = s;
        if (config.type === 'moe') {
            console.warn(`MoE layer '${s}' specified without expert count. Defaulting to 8 total experts.`);
            config.numExperts = 8;
        }
    }
    if (!['dense', 'moe'].includes(config.type)) {
        console.warn(`Unknown layer type: ${config.type}. Treating as dense.`);
        config.type = 'dense';
    }
    return config;
}

function getLayerSignature(layerConfig, hiddenSize, numAttentionHeads, intermediateSizeDense, intermediateSizeMoe, sharedExperts, qkNorm, layerNormStyle, dropoutRate) {
    let signature = `${hiddenSize}-${numAttentionHeads}-${qkNorm}-${layerNormStyle}-${dropoutRate > 0}`;
    if (layerConfig.type === 'dense') {
        signature += `-${layerConfig.type}-${intermediateSizeDense}`;
    } else if (layerConfig.type === 'moe') {
        const sparseExperts = Math.max(0, (layerConfig.numExperts || 0) - sharedExperts);
        signature += `-${layerConfig.type}-${intermediateSizeMoe}-total:${layerConfig.numExperts}-sparse:${sparseExperts}-shared:${sharedExperts}`;
    }
    return signature;
}

function addVisNode(nodes, id, label, options = {}) {
    const defaultOptions = {
        shape: 'box',
        margin: { top: 8, right: 12, bottom: 8, left: 12 },
        font: { size: 10, multi: 'html', align: 'center', color: '#333' },
        borderWidth: 1,
        widthConstraint: { maximum: 180 }, // Max width for nodes
    };
    nodes.push({ id, label: label.replace(/\n/g, '<br>'), ...defaultOptions, ...options }); // Replace \n with <br> for HTML labels
}

function addVisEdge(edges, from, to, options = {}) {
    const defaultOptions = {
        id: getNextVisEdgeId(),
        arrows: {to: {enabled: true, scaleFactor: 0.8}},
        smooth: { type: 'cubicBezier', forceDirection: 'vertical', roundness: 0.4 },
        color: { color: '#607D8B', highlight: '#455A64', hover: '#546E7A' }, // Muted edge color
        font: { size: 9, align: 'middle', color: '#4A4A4A', strokeWidth: 0 },
    };
    edges.push({ from, to, ...defaultOptions, ...options });
}

function addDropoutNodeIfNeeded(nodes, edges, parentNodeId, currentLevel, dropoutRate, commonXOffset = 0) {
    if (dropoutRate > 0) {
        const dropoutNodeId = getNextVisNodeId();
        addVisNode(nodes, dropoutNodeId, `Dropout\n(p=${dropoutRate})`, {
            level: currentLevel,
            x: commonXOffset,
            color: { background: '#E1BEE7', border: '#BA68C8' } // Light Purple
        });
        addVisEdge(edges, parentNodeId, dropoutNodeId);
        return { outputNodeId: dropoutNodeId, nextLevel: currentLevel + 1 };
    }
    return { outputNodeId: parentNodeId, nextLevel: currentLevel }; // No change if no dropout
}


// --- MHA Block Builder ---
function buildMhaBlock(nodes, edges, parentNodeId, startLevel, hiddenSize, numAttentionHeads, positionalEmbeddingType, enableQKNorm, dropoutRate, commonXOffset = 0) {
    const headDim = Math.floor(hiddenSize / numAttentionHeads); // Ensure integer
    let currentLevel = startLevel;
    const mhaInputNodeId = parentNodeId;

    const headsToShow = Math.min(numAttentionHeads, 2);
    const headOutputNodes = [];
    const headXSpacing = 220; // Increased spacing for QK Norm

    for (let i = 0; i < headsToShow; i++) {
        const headX = commonXOffset + i * headXSpacing - ((headsToShow -1) * headXSpacing / 2); // Center the heads group
        let qkvLevel = currentLevel;

        let qLabel = `Q Lin (${hiddenSize}→${headDim})`;
        let kLabel = `K Lin (${hiddenSize}→${headDim})`;
        if (positionalEmbeddingType === 'rotary') {
            qLabel += "\n+RoPE";
            kLabel += "\n+RoPE";
        }

        const qNodeId = getNextVisNodeId();
        addVisNode(nodes, qNodeId, qLabel, { level: qkvLevel, x: headX - 60, color: { background: '#A9D0F5', border: '#64B5F6' } });
        addVisEdge(edges, mhaInputNodeId, qNodeId);

        const kNodeId = getNextVisNodeId();
        addVisNode(nodes, kNodeId, kLabel, { level: qkvLevel, x: headX + 0, color: { background: '#A9D0F5', border: '#64B5F6' } });
        addVisEdge(edges, mhaInputNodeId, kNodeId);

        const vNodeId = getNextVisNodeId();
        addVisNode(nodes, vNodeId, `V Lin (${hiddenSize}→${headDim})`, { level: qkvLevel, x: headX + 60, color: { background: '#A9D0F5', border: '#64B5F6' } });
        addVisEdge(edges, mhaInputNodeId, vNodeId);

        let lastQPathNode = qNodeId;
        let lastKPathNode = kNodeId;
        let qkNormLevel = qkvLevel + 1;

        if (enableQKNorm) {
            const qkNormNodeId = getNextVisNodeId();
            addVisNode(nodes, qkNormNodeId, 'QK Norm\n(e.g., RMSNorm)', { level: qkNormLevel, x: headX - 30 , color: { background: '#81C7FF', border: '#039BE5' }});
            addVisEdge(edges, qNodeId, qkNormNodeId, {label: 'Q'});
            addVisEdge(edges, kNodeId, qkNormNodeId, {label: 'K'});
            lastQPathNode = qkNormNodeId; // Q output from QK Norm
            lastKPathNode = qkNormNodeId; // K output from QK Norm (conceptually, norm applied to both, then used)
            qkNormLevel++;
        }

        const sdpaSoftmaxNodeId = getNextVisNodeId();
        addVisNode(nodes, sdpaSoftmaxNodeId, 'Scaled Dot-Prod Attn\n+ Softmax', { level: qkNormLevel, x: headX, color: { background: '#81C7FF', border: '#039BE5' } });
        addVisEdge(edges, lastQPathNode, sdpaSoftmaxNodeId, { label: enableQKNorm ? 'Normed Q' : 'Q' });
        addVisEdge(edges, lastKPathNode, sdpaSoftmaxNodeId, { label: enableQKNorm ? 'Normed K' : 'K' });
        addVisEdge(edges, vNodeId, sdpaSoftmaxNodeId, { label: 'V' });
        headOutputNodes.push(sdpaSoftmaxNodeId);
        currentLevel = Math.max(currentLevel, qkNormLevel +1); // Ensure currentLevel is past the deepest part of a head
    }


    if (numAttentionHeads > headsToShow) {
        const dotsNodeId = getNextVisNodeId();
        addVisNode(nodes, dotsNodeId, `... ${numAttentionHeads - headsToShow} more ...\n(Identical Heads)`, {
            level: currentLevel -1 , // Align with output of shown heads
            x: commonXOffset + headsToShow * headXSpacing - ((headsToShow -1) * headXSpacing / 2) - (headXSpacing/2) , // Position after last shown head
            shape: 'text', font: {size: 10}, color: { background: 'transparent', border: 'transparent' }
        });
        addVisEdge(edges, mhaInputNodeId, dotsNodeId, {dashes: true, arrows: {to: {enabled: false}}});
        headOutputNodes.push(dotsNodeId);
    }

    const concatNodeId = getNextVisNodeId();
    addVisNode(nodes, concatNodeId, 'Concat Heads', { level: currentLevel, x: commonXOffset, color: { background: '#90CAF9', border: '#64B5F6' } });
    headOutputNodes.forEach(hid => addVisEdge(edges, hid, concatNodeId));
    currentLevel++;

    const outputLinearNodeId = getNextVisNodeId();
    addVisNode(nodes, outputLinearNodeId, `Output Linear\n(${numAttentionHeads * headDim}→${hiddenSize})`, { level: currentLevel, x: commonXOffset, color: { background: '#90CAF9', border: '#64B5F6' } });
    addVisEdge(edges, concatNodeId, outputLinearNodeId);
    currentLevel++;

    return { outputNodeId: outputLinearNodeId, nextLevel: currentLevel };
}

function buildExpertSubGraph(nodes, edges, parentNodeId, expertLevel, expertIdx, hiddenSize, intermediateSize, expertType, commonXOffset = 0, expertXSpacing = 220) {
    const expertPrefix = `${expertType}_expert${expertIdx}_${getNextVisNodeId()}`;
    const expertX = commonXOffset + expertIdx * expertXSpacing;

    const expLinear1Id = `${expertPrefix}_lin1`;
    addVisNode(nodes, expLinear1Id, `${expertType} Expert ${expertIdx + 1}\nLinear 1\n(${hiddenSize}→${intermediateSize})`, { level: expertLevel, x: expertX, color: { background: expertType === 'Shared' ? '#C8E6C9' : '#FFCCBC', border: expertType === 'Shared' ? '#81C784' : '#FF8A65' } });
    addVisEdge(edges, parentNodeId, expLinear1Id);

    const expActivationId = `${expertPrefix}_act`;
    addVisNode(nodes, expActivationId, 'Activation', { level: expertLevel + 1, x: expertX, shape: 'ellipse', color: { background: expertType === 'Shared' ? '#A5D6A7' : '#FFE0B2', border: expertType === 'Shared' ? '#66BB6A' : '#FF8A65' } });
    addVisEdge(edges, expLinear1Id, expActivationId);

    const expLinear2Id = `${expertPrefix}_lin2`;
    addVisNode(nodes, expLinear2Id, `Linear 2\n(${intermediateSize}→${hiddenSize})`, { level: expertLevel + 2, x: expertX, color: { background: expertType === 'Shared' ? '#C8E6C9' : '#FFCCBC', border: expertType === 'Shared' ? '#81C784' : '#FF8A65' } });
    addVisEdge(edges, expActivationId, expLinear2Id);
    return expLinear2Id; // Return the output node of this expert
}


// --- FFN Block Builder ---
function buildFfnBlock(nodes, edges, parentNodeId, startLevel, hiddenSize, intermediateSize, ffnConfig, numSharedExperts, commonXOffset = 0) {
    const ffnType = ffnConfig.type;
    let currentLevel = startLevel;
    const expertXSpacing = 220; // Spacing between horizontal experts

    if (ffnType === 'dense') {
        const linear1Id = getNextVisNodeId();
        addVisNode(nodes, linear1Id, `Linear\n(${hiddenSize}→${intermediateSize})`, { level: currentLevel, x: commonXOffset, color: { background: '#FFF59D', border: '#FFEE58' } });
        addVisEdge(edges, parentNodeId, linear1Id);
        currentLevel++;

        const activationId = getNextVisNodeId();
        addVisNode(nodes, activationId, 'Activation\n(e.g., GELU)', { level: currentLevel, x: commonXOffset, shape: 'ellipse', color: { background: '#FFF9C4', border: '#FFEE58' } });
        addVisEdge(edges, linear1Id, activationId);
        currentLevel++;

        const linear2Id = getNextVisNodeId();
        addVisNode(nodes, linear2Id, `Linear\n(${intermediateSize}→${hiddenSize})`, { level: currentLevel, x: commonXOffset, color: { background: '#FFF59D', border: '#FFEE58' } });
        addVisEdge(edges, activationId, linear2Id);
        currentLevel++;
        return { outputNodeId: linear2Id, nextLevel: currentLevel };
    } else if (ffnType === 'moe') {
        const totalExpertsInConfig = ffnConfig.numExperts || 0;
        const sparseExpertsCount = Math.max(0, totalExpertsInConfig - numSharedExperts);
        const sparseExpertsToDraw = Math.min(sparseExpertsCount, 2);
        const sharedExpertsToDraw = Math.min(numSharedExperts, 2); // Also draw up to 2 shared experts fully

        const expertOutputNodes = [];
        let maxExpertDepthLevel = currentLevel;

        // --- Shared Experts ---
        // Position shared experts to the left of sparse experts or centered if no sparse
        const sharedExpertsStartX = commonXOffset - (sparseExpertsToDraw > 0 ? (sparseExpertsToDraw * expertXSpacing / 2 + expertXSpacing /2) : 0) - ((sharedExpertsToDraw -1 ) * expertXSpacing /2) ;

        for (let i = 0; i < sharedExpertsToDraw; i++) {
            const expertOutNodeId = buildExpertSubGraph(nodes, edges, parentNodeId, currentLevel, i, hiddenSize, intermediateSize, 'Shared', sharedExpertsStartX, expertXSpacing);
            expertOutputNodes.push(expertOutNodeId);
        }
        if (sharedExpertsToDraw > 0) maxExpertDepthLevel = Math.max(maxExpertDepthLevel, currentLevel + 2);

        if (numSharedExperts > sharedExpertsToDraw) {
            const dotsSharedNodeId = getNextVisNodeId();
            addVisNode(nodes, dotsSharedNodeId, `... ${numSharedExperts - sharedExpertsToDraw} more ...\n(Shared Experts)`, {
                level: currentLevel, x: sharedExpertsStartX + sharedExpertsToDraw * expertXSpacing,
                shape: 'text', font: {size: 10}, color: { background: 'transparent', border: 'transparent' }
            });
            addVisEdge(edges, parentNodeId, dotsSharedNodeId, {dashes: true, arrows: {to: {enabled: false}}});
            expertOutputNodes.push(dotsSharedNodeId);
        }

        // --- Sparse Experts (if any) ---
        let routerId = parentNodeId; // If no shared experts, router comes from main parent.
                                  // If shared, router is a separate concept for sparse.
        if (sparseExpertsCount > 0) {
            routerId = getNextVisNodeId(); // Create a distinct router for sparse ones
            addVisNode(nodes, routerId, `Router (Gating)\nSelects Top-K Sparse`, {
                level: maxExpertDepthLevel + (expertOutputNodes.length > 0 ? 1: 0), // Place router after any shared experts if they exist or at current if not
                x: commonXOffset, shape: 'diamond', color: { background: '#FFAB91', border: '#FF8A65' }
            });
            addVisEdge(edges, parentNodeId, routerId); // Main input to router
            maxExpertDepthLevel = Math.max(maxExpertDepthLevel, parseInt(nodes.find(n=>n.id === routerId).level));
        }

        const sparseExpertsStartX = commonXOffset + (numSharedExperts > 0 ? (numSharedExperts * expertXSpacing / 2 + expertXSpacing /2) : 0) - ((sparseExpertsToDraw-1) * expertXSpacing /2);
        let sparseExpertStartLevel = routerId === parentNodeId ? currentLevel : maxExpertDepthLevel + 1; // Start sparse experts after router

        for (let i = 0; i < sparseExpertsToDraw; i++) {
            const expertOutNodeId = buildExpertSubGraph(nodes, edges, routerId, sparseExpertStartLevel, i, hiddenSize, intermediateSize, 'Sparse', sparseExpertsStartX, expertXSpacing);
            expertOutputNodes.push(expertOutNodeId);
        }
        if (sparseExpertsToDraw > 0) maxExpertDepthLevel = Math.max(maxExpertDepthLevel, sparseExpertStartLevel + 2);


        if (sparseExpertsCount > sparseExpertsToDraw) {
            const dotsSparseNodeId = getNextVisNodeId();
            addVisNode(nodes, dotsSparseNodeId, `... ${sparseExpertsCount - sparseExpertsToDraw} more ...\n(Sparse Experts)`, {
                level: sparseExpertStartLevel, x: sparseExpertsStartX + sparseExpertsToDraw * expertXSpacing,
                shape: 'text', font: {size: 10}, color: { background: 'transparent', border: 'transparent' }
            });
            addVisEdge(edges, routerId, dotsSparseNodeId, {dashes: true, arrows: {to: {enabled: false}}});
            expertOutputNodes.push(dotsSparseNodeId);
        }

        currentLevel = maxExpertDepthLevel + 1;
        const aggregationId = getNextVisNodeId();
        addVisNode(nodes, aggregationId, 'Aggregation\n(Weighted Sum + Shared)', { level: currentLevel, x: commonXOffset, color: { background: '#FFAB91', border: '#FF8A65' } });
        expertOutputNodes.forEach(outId => addVisEdge(edges, outId, aggregationId));
        currentLevel++;

        return { outputNodeId: aggregationId, nextLevel: currentLevel };
    }
    return { outputNodeId: parentNodeId, nextLevel: currentLevel + 1 }; // Fallback
}


function drawTransformer() {
    visNodeIdCounter = 1; visEdgeIdCounter = 1; // Reset IDs

    const numLayers = parseInt(document.getElementById('numLayers').value);
    const hiddenSize = parseInt(document.getElementById('hiddenSize').value);
    const numAttentionHeads = parseInt(document.getElementById('numAttentionHeads').value);
    const intermediateSizeDense = parseInt(document.getElementById('intermediateSizeDense').value);
    const intermediateSizeMoe = parseInt(document.getElementById('intermediateSizeMoe').value);
    const layerTypesStrRaw = document.getElementById('layerTypes').value;
    const positionalEmbeddingType = document.getElementById('positionalEmbeddingType').value;
    const moeSharedExperts = parseInt(document.getElementById('moeSharedExperts').value) || 0;
    const enableQKNorm = document.getElementById('qkNormalization').checked;
    const layerNormStyle = document.getElementById('layerNormStyle').value; // 'pre_ln' or 'post_ln'
    const dropoutRate = parseFloat(document.getElementById('dropoutRate').value) || 0.0;

    const layerConfigsInput = layerTypesStrRaw.split(',').map(s => s.trim().toLowerCase()).filter(s => s);
    if (layerConfigsInput.length !== numLayers && numLayers > 0) {
        alert(`Error: The number of layer type definitions (${layerConfigsInput.length}) must match total layers (${numLayers}).`);
        return;
    }
    const layerConfigs = layerConfigsInput.map(parseLayerConfigString);

    const nodes = [];
    const edges = [];
    let currentLevel = 0;
    let commonXOffset = 0; // Can be used to center the main vertical flow

    // --- Optional RoPE Annotation Node ---
    if (positionalEmbeddingType === 'rotary') {
        const ropeInfoNodeId = getNextVisNodeId();
        const ropeLabel = `Rotary Positional Embedding (RoPE) Info:
- Applied directly to Query & Key vectors in Attention.
- Rotates embedding vectors based on absolute position 'm'.
- Achieved by element-wise multiplication with complex numbers representing rotation:
  x'_k = x_k * e^(i * m * θ_k)
  where θ_k are pre-defined frequencies.
- Effectively mixes pairs of features:
  [x_i, x_{i+1}] transformed by rotation matrix.`;
        addVisNode(nodes, ropeInfoNodeId, ropeLabel, {
            level: 0, x: 400, // Position to the side
            color: { background: '#FFF9C4', border: '#FFEE58' }, // Light Yellow, like a sticky note
            shape: 'note', // Use 'note' shape if available/supported, else box
            widthConstraint: { maximum: 250 }, // Wider for more text
            font: {align: 'left', size: 9}
        });
        currentLevel = 1; // Start main graph below RoPE info
    }


    // --- 1. Input Embedding ---
    let embLabel = `Token Embeddings\n(Vocab → ${hiddenSize})`;
    const tokenEmbNodeId = getNextVisNodeId();
    addVisNode(nodes, tokenEmbNodeId, embLabel, { level: currentLevel, x: commonXOffset -50, color: { background: '#E0E0E0', border: '#BDBDBD' } });
    let lastProcessedNode = tokenEmbNodeId;

    if (positionalEmbeddingType !== 'none' && positionalEmbeddingType !== 'rotary' && positionalEmbeddingType !== 'alibi') {
        let posEmbLabel = `Positional Embeddings\n(${hiddenSize})`;
        if (positionalEmbeddingType === 'sinusoidal_absolute') posEmbLabel = `Sinusoidal Absolute PE\n(${hiddenSize})`;
        else posEmbLabel = `Learned Absolute PE\n(${hiddenSize})`;

        const posEmbNodeId = getNextVisNodeId();
        addVisNode(nodes, posEmbNodeId, posEmbLabel, { level: currentLevel, x: commonXOffset + 50, color: { background: '#E0E0E0', border: '#BDBDBD' } });

        const sumEmbNodeId = getNextVisNodeId();
        addVisNode(nodes, sumEmbNodeId, '+', { level: currentLevel + 1, x: commonXOffset, shape: 'circle', size: 15, color: { background: '#EEEEEE', border: '#BDBDBD' } });
        addVisEdge(edges, tokenEmbNodeId, sumEmbNodeId);
        addVisEdge(edges, posEmbNodeId, sumEmbNodeId);
        lastProcessedNode = sumEmbNodeId;
        currentLevel++;
    }
    currentLevel++; // Space after embeddings / sum

    // Dropout after embeddings
    let dropoutResult = addDropoutNodeIfNeeded(nodes, edges, lastProcessedNode, currentLevel, dropoutRate, commonXOffset);
    lastProcessedNode = dropoutResult.outputNodeId;
    currentLevel = dropoutResult.nextLevel;


    // --- Initial LayerNorm (Only for Post-LN style, or if it's a global first LN) ---
    // For Pre-LN, the first LN is part of the first layer block.
    // Let's assume if Pre-LN, the input to the first block is `lastProcessedNode`
    // and the block itself starts with LN.
    // If Post-LN, we might need an initial LN before the loop.
    // For simplicity with Pre-LN as default, we'll proceed, and each Pre-LN block handles its LN.
    // If layerNormStyle is 'post_ln', an initial LN might be drawn here if not handled by loop structure.

    let lastBlockOutputNodeId = lastProcessedNode;


    // --- 2. Transformer Layers ---
    for (let layerIdx = 0; layerIdx < numLayers; layerIdx++) {
        const currentLayerConfig = layerConfigs[layerIdx];
        // Layer Grouping/Repetition Info (No separate title node)
        // This info can be prepended to the first node of the layer, e.g., the first LayerNorm.
        let blockRepetitionInfo = `L${layerIdx + 1} `; // Basic layer info
        // (Skipping full repetition check for now to simplify, focus on internal structure)

        let currentLayerInput = lastBlockOutputNodeId;
        let residualConnectionStart = currentLayerInput;

        // --- Pre-MHA LayerNorm (Common for both Pre-LN and Post-LN starts a sublayer) ---
        const lnPreMhaId = getNextVisNodeId();
        addVisNode(nodes, lnPreMhaId, `${blockRepetitionInfo}LayerNorm\n(Pre-MHA)`, { level: currentLevel, x: commonXOffset, color: { background: '#BBDEFB', border: '#90CAF9' } });
        addVisEdge(edges, currentLayerInput, lnPreMhaId);
        let mhaInput = lnPreMhaId;
        if (layerNormStyle === 'pre_ln') {
            residualConnectionStart = currentLayerInput; // For Pre-LN, residual starts *before* this LN
        } else { // Post-LN style
            residualConnectionStart = lnPreMhaId; // Or after, depending on exact Post-LN variant
        }


        // --- MHA Block ---
        const mhaResult = buildMhaBlock(nodes, edges, mhaInput, currentLevel + 1, hiddenSize, numAttentionHeads, positionalEmbeddingType, enableQKNorm, dropoutRate, commonXOffset);
        let mhaOutput = mhaResult.outputNodeId;
        currentLevel = mhaResult.nextLevel;

        // Dropout after MHA
        dropoutResult = addDropoutNodeIfNeeded(nodes, edges, mhaOutput, currentLevel, dropoutRate, commonXOffset);
        mhaOutput = dropoutResult.outputNodeId;
        currentLevel = dropoutResult.nextLevel;


        // --- Add & Norm after MHA (Post-LN) or Add + next LN (Pre-LN) ---
        const addMhaResId = getNextVisNodeId();
        let postMhaProcessingNode;
        if (layerNormStyle === 'post_ln') {
            addVisNode(nodes, addMhaResId, 'Add + LayerNorm\n(Post-MHA)', { level: currentLevel, x: commonXOffset, color: { background: '#BBDEFB', border: '#90CAF9' } });
            addVisEdge(edges, mhaOutput, addMhaResId);
            addVisEdge(edges, residualConnectionStart, addMhaResId, { dashes: true, label: 'Res', smooth: { type: 'curvedCW', roundness: 0.3 }});
            postMhaProcessingNode = addMhaResId;
        } else { // Pre-LN style
            addVisNode(nodes, addMhaResId, 'Add\n(MHA Res)', { level: currentLevel, x: commonXOffset, shape:'circle', size:15, color: { background: '#EEEEEE', border: '#BDBDBD' }});
            addVisEdge(edges, mhaOutput, addMhaResId);
            addVisEdge(edges, residualConnectionStart, addMhaResId, { dashes: true, label: 'Res', smooth: { type: 'curvedCW', roundness: 0.3 }});
            postMhaProcessingNode = addMhaResId; // This is now input to Pre-FFN LN
        }
        currentLevel++;
        residualConnectionStart = postMhaProcessingNode; // Update for next residual path (for FFN)


        // --- Pre-FFN LayerNorm (For Pre-LN style) ---
        let ffnInput = postMhaProcessingNode;
        if (layerNormStyle === 'pre_ln') {
            const lnPreFfnId = getNextVisNodeId();
            addVisNode(nodes, lnPreFfnId, 'LayerNorm\n(Pre-FFN)', { level: currentLevel, x: commonXOffset, color: { background: '#BBDEFB', border: '#90CAF9' } });
            addVisEdge(edges, postMhaProcessingNode, lnPreFfnId);
            ffnInput = lnPreFfnId;
            residualConnectionStart = postMhaProcessingNode; // Residual for FFN starts *before* this LN in Pre-LN
            currentLevel++;
        }

        // --- FFN Block ---
        const ffnResult = buildFfnBlock(nodes, edges, ffnInput, currentLevel, hiddenSize, currentLayerConfig.type === 'dense' ? intermediateSizeDense : intermediateSizeMoe, currentLayerConfig, moeSharedExperts, commonXOffset);
        let ffnOutput = ffnResult.outputNodeId;
        currentLevel = ffnResult.nextLevel;

        // Dropout after FFN
        dropoutResult = addDropoutNodeIfNeeded(nodes, edges, ffnOutput, currentLevel, dropoutRate, commonXOffset);
        ffnOutput = dropoutResult.outputNodeId;
        currentLevel = dropoutResult.nextLevel;


        // --- Add & Norm after FFN (Post-LN) or just Add (Pre-LN) ---
        const addFfnResId = getNextVisNodeId();
        if (layerNormStyle === 'post_ln') {
            addVisNode(nodes, addFfnResId, 'Add + LayerNorm\n(Post-FFN)', { level: currentLevel, x: commonXOffset, color: { background: '#BBDEFB', border: '#90CAF9' } });
            addVisEdge(edges, ffnOutput, addFfnResId);
            addVisEdge(edges, residualConnectionStart, addFfnResId, { dashes: true, label: 'Res', smooth: { type: 'curvedCW', roundness: 0.3 }});
            lastBlockOutputNodeId = addFfnResId;
        } else { // Pre-LN style
            addVisNode(nodes, addFfnResId, 'Add\n(FFN Res)', { level: currentLevel, x: commonXOffset, shape:'circle', size:15, color: { background: '#EEEEEE', border: '#BDBDBD' }});
            addVisEdge(edges, ffnOutput, addFfnResId);
            addVisEdge(edges, residualConnectionStart, addFfnResId, { dashes: true, label: 'Res', smooth: { type: 'curvedCW', roundness: 0.3 }});
            lastBlockOutputNodeId = addFfnResId; // This output goes to the next layer's Pre-MHA LN
        }
        currentLevel++;
    } // End of layers loop

    // --- Final LayerNorm (if Pre-LN style and layers exist) ---
    if (layerNormStyle === 'pre_ln' && numLayers > 0) {
        const finalLNId = getNextVisNodeId();
        addVisNode(nodes, finalLNId, 'LayerNorm\n(Final)', { level: currentLevel, x: commonXOffset, color: { background: '#BBDEFB', border: '#90CAF9' } });
        addVisEdge(edges, lastBlockOutputNodeId, finalLNId);
        lastBlockOutputNodeId = finalLNId;
        currentLevel++;
    }

    // --- Output LM Head (if model has layers) ---
    let finalModelOutputNode = lastBlockOutputNodeId;
    if (numLayers > 0) {
        const lmHeadId = getNextVisNodeId();
        addVisNode(nodes, lmHeadId, `LM Head Linear\n(${hiddenSize} → Vocab Size)`, { level: currentLevel, x: commonXOffset, color: { background: '#CFD8DC', border: '#90A4AE' } });
        addVisEdge(edges, lastBlockOutputNodeId, lmHeadId);
        currentLevel++;

        const softmaxOutputId = getNextVisNodeId();
        addVisNode(nodes, softmaxOutputId, 'Softmax\n(Logits → Probs)', { level: currentLevel, x: commonXOffset, color: { background: '#CFD8DC', border: '#90A4AE' } });
        addVisEdge(edges, lmHeadId, softmaxOutputId);
        finalModelOutputNode = softmaxOutputId;
        currentLevel++;
    }


    // --- Training and Inference Branches ---
    if (finalModelOutputNode && numLayers > 0) { // Only if there's an actual model output
        // Training Branch
        const lossNodeId = getNextVisNodeId();
        addVisNode(nodes, lossNodeId, 'Training: Cross-Entropy Loss\n(vs Target Tokens)', {
            level: currentLevel, x: commonXOffset - 150, // To the left
            color: { background: '#C5CAE9', border: '#7986CB' } // Indigo
        });
        addVisEdge(edges, finalModelOutputNode, lossNodeId);

        // Inference Branch
        const samplingNodeId = getNextVisNodeId();
        addVisNode(nodes, samplingNodeId, 'Inference: Sampling Logic\n(Greedy, Top-K, Top-P, Temp)', {
            level: currentLevel, x: commonXOffset + 150, // To the right
            color: { background: '#B2DFDB', border: '#4DB6AC' } // Teal
        });
        addVisEdge(edges, finalModelOutputNode, samplingNodeId);
        currentLevel++;
    }


    // --- Vis.js Network ---
    const container = document.getElementById('mynetwork');
    const data = {
        nodes: new vis.DataSet(nodes),
        edges: new vis.DataSet(edges),
    };
    const options = {
        layout: {
            hierarchical: {
                enabled: true, direction: 'UD', sortMethod: 'directed',
                levelSeparation: 100, nodeSpacing: 130, treeSpacing: 180,
                blockShifting: true, edgeMinimization: true, parentCentralization: true,
            },
        },
        physics: { enabled: false },
        nodes: { font: { color: '#333' }, widthConstraint: { maximum: 200 } },
        edges: { width: 1.5, selectionWidth: 2, hoverWidth: 0.5 },
        interaction: {
            hover: true, tooltipDelay: 200, dragNodes: true, dragView: true, zoomView: true,
            navigationButtons: true, keyboard: true,
        },
    };

    const network = new vis.Network(container, data, options);
    network.on("stabilizationIterationsDone", () => {
        network.setOptions( { physics: false } ); // Ensure physics stays off after stabilization
         try { network.fit({animation: {duration: 300, easingFunction: 'linear'}}); } catch(e) {}
    });
     setTimeout(() => { // Fallback fit
         try { network.fit({animation: {duration: 300, easingFunction: 'linear'}}); } catch(e) {}
    }, 500);
}

window.addEventListener('load', () => {
    const numLayersInput = document.getElementById('numLayers');
    const layerTypesInput = document.getElementById('layerTypes');
    if (numLayersInput && layerTypesInput) {
        const numLayers = parseInt(numLayersInput.value);
        const types = layerTypesInput.value.split(',').map(s=>s.trim()).filter(s=>s);
        if (types.length !== numLayers && numLayers > 0) {
            let defaultTypes = [];
            for(let i=0; i<numLayers; i++) defaultTypes.push(i % 2 === 0 ? 'dense' : 'moe:8');
            layerTypesInput.value = defaultTypes.join(',');
        } else if (numLayers === 0) layerTypesInput.value = "";
    }
     // Check if moe_specific_controls should be visible
    const updateMoeControlsVisibility = () => {
        const layerTypesStr = document.getElementById('layerTypes').value;
        const moeSpecificControls = document.getElementById('moe_specific_controls');
        if (layerTypesStr.includes('moe')) {
            moeSpecificControls.style.display = 'block';
        } else {
            moeSpecificControls.style.display = 'none';
        }
    };
    layerTypesInput.addEventListener('input', updateMoeControlsVisibility);
    updateMoeControlsVisibility(); // Initial check

    // drawTransformer(); // Optionally draw on load
});