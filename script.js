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

function getLayerSignature(layerConfig, hiddenSize, numAttentionHeads, intermediateSizeDense, intermediateSizeMoe, sharedExperts) {
    if (layerConfig.type === 'dense') {
        return `${layerConfig.type}-${hiddenSize}-${numAttentionHeads}-${intermediateSizeDense}`;
    } else if (layerConfig.type === 'moe') {
        // Signature should depend on total sparse experts and shared experts if they affect structure
        const sparseExperts = Math.max(0, (layerConfig.numExperts || 0) - sharedExperts);
        return `${layerConfig.type}-${hiddenSize}-${numAttentionHeads}-${intermediateSizeMoe}-total:${layerConfig.numExperts}-sparse:${sparseExperts}-shared:${sharedExperts}`;
    }
    return 'unknown';
}

function addVisNode(nodes, id, label, options = {}) {
    const defaultOptions = {
        shape: 'box',
        margin: { top: 8, right: 12, bottom: 8, left: 12 }, // Increased margin for text
        font: { size: 10, multi: 'html', align: 'center' },
        borderWidth: 1,
    };
    nodes.push({ id, label, ...defaultOptions, ...options });
}

function addVisEdge(edges, from, to, options = {}) {
    const defaultOptions = {
        id: getNextVisEdgeId(),
        arrows: 'to',
        smooth: { type: 'cubicBezier', forceDirection: 'vertical', roundness: 0.4 },
        color: { color: '#848484', highlight: '#575757' },
        font: { size: 9, align: 'middle', color: '#555555', strokeWidth: 0 },
    };
    edges.push({ from, to, ...defaultOptions, ...options });
}

// --- MHA Block Builder ---
function buildMhaBlock(nodes, edges, parentNodeId, startLevel, hiddenSize, numAttentionHeads, positionalEmbeddingType) {
    const headDim = hiddenSize / numAttentionHeads;
    let currentLevel = startLevel;
    const mhaInputNodeId = parentNodeId; // Input to Q,K,V generation

    const headsToShow = Math.min(numAttentionHeads, 2);
    const headOutputNodes = [];

    // Horizontal layout for head representations
    for (let i = 0; i < headsToShow; i++) {
        const headPrefix = `head${i}_${getNextVisNodeId()}`; // Unique prefix for nodes in this head
        let qkvLevel = currentLevel;

        let qLabel = `Q Lin (${hiddenSize}→${headDim})`;
        let kLabel = `K Lin (${hiddenSize}→${headDim})`;
        if (positionalEmbeddingType === 'rotary') {
            qLabel += "\n+RoPE";
            kLabel += "\n+RoPE";
        }

        const qNodeId = `${headPrefix}_q`;
        addVisNode(nodes, qNodeId, qLabel, { level: qkvLevel, x: i * 250, color: { background: '#A9D0F5', border: '#64B5F6' } });
        addVisEdge(edges, mhaInputNodeId, qNodeId);

        const kNodeId = `${headPrefix}_k`;
        addVisNode(nodes, kNodeId, kLabel, { level: qkvLevel, x: i * 250 + 80, color: { background: '#A9D0F5', border: '#64B5F6' } });
        addVisEdge(edges, mhaInputNodeId, kNodeId); // Also from MHA input

        const vNodeId = `${headPrefix}_v`;
        addVisNode(nodes, vNodeId, `V Lin (${hiddenSize}→${headDim})`, { level: qkvLevel, x: i * 250 + 160, color: { background: '#A9D0F5', border: '#64B5F6' } });
        addVisEdge(edges, mhaInputNodeId, vNodeId); // Also from MHA input

        const sdpaSoftmaxNodeId = `${headPrefix}_sdpa_softmax`;
        addVisNode(nodes, sdpaSoftmaxNodeId, 'Scaled Dot-Prod Attn\n+ Softmax', { level: qkvLevel + 1, x: i * 250 + 80, color: { background: '#81C7FF', border: '#039BE5' } }); // Darker blue
        addVisEdge(edges, qNodeId, sdpaSoftmaxNodeId, { label: 'Q' });
        addVisEdge(edges, kNodeId, sdpaSoftmaxNodeId, { label: 'K' });
        addVisEdge(edges, vNodeId, sdpaSoftmaxNodeId, { label: 'V' });
        headOutputNodes.push(sdpaSoftmaxNodeId);
    }
    currentLevel += 2; // After QKV and SDPA

    let lastHorizontalNodeConnector = headOutputNodes[headOutputNodes.length - 1];

    if (numAttentionHeads > headsToShow) {
        const dotsNodeId = getNextVisNodeId();
        addVisNode(nodes, dotsNodeId, `... ${numAttentionHeads - headsToShow} more ...\n(Identical Heads)`, {
            level: currentLevel -1 , // Align with output of shown heads
            x: headsToShow * 250,
            shape: 'text',
            font: {size: 10},
            color: { background: 'transparent', border: 'transparent' }
        });
        // Conceptually, input also goes to these "..." heads
        addVisEdge(edges, mhaInputNodeId, dotsNodeId, {dashes: true, arrows: {to: {enabled: false}}});
        headOutputNodes.push(dotsNodeId); // This "..." also feeds into concat
        lastHorizontalNodeConnector = dotsNodeId;
    }


    const concatNodeId = getNextVisNodeId();
    addVisNode(nodes, concatNodeId, 'Concat Heads', { level: currentLevel, color: { background: '#90CAF9', border: '#64B5F6' } });
    headOutputNodes.forEach(hid => addVisEdge(edges, hid, concatNodeId));
    currentLevel++;

    const outputLinearNodeId = getNextVisNodeId();
    addVisNode(nodes, outputLinearNodeId, `Output Linear\n(${numAttentionHeads * headDim}→${hiddenSize})`, { level: currentLevel, color: { background: '#90CAF9', border: '#64B5F6' } });
    addVisEdge(edges, concatNodeId, outputLinearNodeId);
    currentLevel++;

    return { outputNodeId: outputLinearNodeId, nextLevel: currentLevel };
}

// --- FFN Block Builder ---
function buildFfnBlock(nodes, edges, parentNodeId, startLevel, hiddenSize, intermediateSize, ffnConfig, numSharedExperts) {
    const ffnType = ffnConfig.type;
    let currentLevel = startLevel;

    if (ffnType === 'dense') {
        const linear1Id = getNextVisNodeId();
        addVisNode(nodes, linear1Id, `Linear\n(${hiddenSize}→${intermediateSize})`, { level: currentLevel, color: { background: '#FFF59D', border: '#FFEE58' } });
        addVisEdge(edges, parentNodeId, linear1Id);
        currentLevel++;

        const activationId = getNextVisNodeId();
        addVisNode(nodes, activationId, 'Activation\n(e.g., GELU)', { level: currentLevel, shape: 'ellipse', color: { background: '#FFF9C4', border: '#FFEE58' } });
        addVisEdge(edges, linear1Id, activationId);
        currentLevel++;

        const linear2Id = getNextVisNodeId();
        addVisNode(nodes, linear2Id, `Linear\n(${intermediateSize}→${hiddenSize})`, { level: currentLevel, color: { background: '#FFF59D', border: '#FFEE58' } });
        addVisEdge(edges, activationId, linear2Id);
        currentLevel++;
        return { outputNodeId: linear2Id, nextLevel: currentLevel };
    } else if (ffnType === 'moe') {
        const totalExpertsInConfig = ffnConfig.numExperts || 0;
        const sparseExpertsCount = Math.max(0, totalExpertsInConfig - numSharedExperts);
        const expertsToDraw = Math.min(sparseExpertsCount, 2); // Draw up to 2 sparse experts
        const expertOutputNodes = [];

        const routerId = getNextVisNodeId();
        addVisNode(nodes, routerId, `Router (Gating)\nSelects Top-K Sparse`, { level: currentLevel, shape: 'diamond', color: { background: '#FFAB91', border: '#FF8A65' } });
        addVisEdge(edges, parentNodeId, routerId);
        let routerLevel = currentLevel;
        currentLevel++; // Level for experts

        // Draw Shared Experts (if any) - simplified representation for now
        if (numSharedExperts > 0) {
            const sharedExpertsGroupId = getNextVisNodeId();
            // Represent all shared experts as one block for simplicity, feeding into aggregation.
            // To detail them, loop and draw like sparse experts.
            addVisNode(nodes, sharedExpertsGroupId, `${numSharedExperts} Shared Expert(s)\n(Compute for all tokens)`, {
                level: currentLevel, // Parallel to sparse experts
                x: -250, // Position to the left
                color: { background: '#C8E6C9', border: '#81C784' } // Light Green
            });
            addVisEdge(edges, parentNodeId, sharedExpertsGroupId); // Shared experts also take the main input
            expertOutputNodes.push(sharedExpertsGroupId);
        }


        // Draw Sparse Experts (Horizontal Layout)
        for (let i = 0; i < expertsToDraw; i++) {
            const expertPrefix = `sparse_expert${i}_${getNextVisNodeId()}`;
            let expertLevel = currentLevel; // All experts start at same level

            const expLinear1Id = `${expertPrefix}_lin1`;
            addVisNode(nodes, expLinear1Id, `Sparse Expert ${i + 1}\nLinear 1\n(${hiddenSize}→${intermediateSize})`, { level: expertLevel, x: i * 250, color: { background: '#FFCCBC', border: '#FF8A65' } });
            addVisEdge(edges, routerId, expLinear1Id); // From Router

            const expActivationId = `${expertPrefix}_act`;
            addVisNode(nodes, expActivationId, 'Activation', { level: expertLevel + 1, x: i * 250, shape: 'ellipse', color: { background: '#FFE0B2', border: '#FF8A65' } });
            addVisEdge(edges, expLinear1Id, expActivationId);

            const expLinear2Id = `${expertPrefix}_lin2`;
            addVisNode(nodes, expLinear2Id, `Linear 2\n(${intermediateSize}→${hiddenSize})`, { level: expertLevel + 2, x: i * 250, color: { background: '#FFCCBC', border: '#FF8A65' } });
            addVisEdge(edges, expActivationId, expLinear2Id);
            expertOutputNodes.push(expLinear2Id);
        }
        // Advance currentLevel based on the depth of one expert
        let maxExpertDepthLevel = currentLevel + 2;


        if (sparseExpertsCount > expertsToDraw) {
            const dotsNodeId = getNextVisNodeId();
            addVisNode(nodes, dotsNodeId, `... ${sparseExpertsCount - expertsToDraw} more ...\n(Identical Sparse Experts)`, {
                level: currentLevel, // Align with start of drawn experts
                x: expertsToDraw * 250,
                shape: 'text',
                font: {size: 10},
                color: { background: 'transparent', border: 'transparent' }
            });
            addVisEdge(edges, routerId, dotsNodeId, {dashes: true, arrows: {to: {enabled: false}}}); // From router to "..."
            expertOutputNodes.push(dotsNodeId);
        }
        currentLevel = maxExpertDepthLevel + 1;


        const aggregationId = getNextVisNodeId();
        addVisNode(nodes, aggregationId, 'Aggregation\n(Weighted Sum + Shared)', { level: currentLevel, color: { background: '#FFAB91', border: '#FF8A65' } });
        expertOutputNodes.forEach(outId => addVisEdge(edges, outId, aggregationId));
        currentLevel++;

        return { outputNodeId: aggregationId, nextLevel: currentLevel };
    }
    return { outputNodeId: parentNodeId, nextLevel: currentLevel + 1 }; // Fallback
}


function drawTransformer() {
    visNodeIdCounter = 1; // Reset for re-renders
    visEdgeIdCounter = 1;

    const numLayers = parseInt(document.getElementById('numLayers').value);
    const hiddenSize = parseInt(document.getElementById('hiddenSize').value);
    const numAttentionHeads = parseInt(document.getElementById('numAttentionHeads').value);
    const intermediateSizeDense = parseInt(document.getElementById('intermediateSizeDense').value);
    const intermediateSizeMoe = parseInt(document.getElementById('intermediateSizeMoe').value);
    const layerTypesStrRaw = document.getElementById('layerTypes').value;
    const positionalEmbeddingType = document.getElementById('positionalEmbeddingType').value;
    const moeSharedExperts = parseInt(document.getElementById('moeSharedExperts').value) || 0;

    const layerConfigsInput = layerTypesStrRaw.split(',').map(s => s.trim()).filter(s => s);

    if (layerConfigsInput.length !== numLayers && numLayers > 0) {
        alert(`Error: The number of layer type definitions (${layerConfigsInput.length}) must match the total number of layers (${numLayers}). Please adjust inputs.`);
        return;
    }
     if (numLayers === 0 && layerConfigsInput.length > 0) {
        alert(`Error: Number of layers is 0, but layer types are defined. Please adjust inputs.`);
        return;
    }


    const layerConfigs = layerConfigsInput.map(parseLayerConfigString);

    const nodes = [];
    const edges = [];
    let currentLevel = 0; // Start levels from 0 for tighter top spacing

    // --- 1. Input Embedding ---
    let embLabel = `Token Embeddings\n(Vocab → ${hiddenSize})`;
    const tokenEmbNodeId = getNextVisNodeId();
    addVisNode(nodes, tokenEmbNodeId, embLabel, { level: currentLevel, color: { background: '#E0E0E0', border: '#BDBDBD' } });

    let lastEmbeddingNode = tokenEmbNodeId;

    if (positionalEmbeddingType !== 'none') {
        let posEmbLabel = `Positional Embeddings\n(${hiddenSize})`;
        if (positionalEmbeddingType === 'rotary') {
            posEmbLabel = `Rotary PE (RoPE)\n(Applied in Attention Q/K)`;
        } else if (positionalEmbeddingType === 'alibi') {
            posEmbLabel = `ALiBi PE\n(Bias in Attention Scores)`;
        } else if (positionalEmbeddingType === 'sinusoidal_absolute') {
            posEmbLabel = `Sinusoidal Absolute PE\n(${hiddenSize})`;
        } else { // learned_absolute
            posEmbLabel = `Learned Absolute PE\n(${hiddenSize})`;
        }

        const posEmbNodeId = getNextVisNodeId();
        // If RoPE or ALiBi, it's more of a conceptual note here, as it's applied within attention
        const posEmbColor = (positionalEmbeddingType === 'rotary' || positionalEmbeddingType === 'alibi') ?
                            { background: '#E8EAF6', border: '#C5CAE9' } : // Lighter, distinct color for conceptual PEs
                            { background: '#E0E0E0', border: '#BDBDBD' };
        addVisNode(nodes, posEmbNodeId, posEmbLabel, { level: currentLevel, color: posEmbColor }); // Same level

        // If not RoPE/ALiBi, sum them
        if (positionalEmbeddingType !== 'rotary' && positionalEmbeddingType !== 'alibi') {
            currentLevel++;
            const sumEmbNodeId = getNextVisNodeId();
            addVisNode(nodes, sumEmbNodeId, '+', { level: currentLevel, shape: 'circle', size: 15, color: { background: '#EEEEEE', border: '#BDBDBD' } });
            addVisEdge(edges, tokenEmbNodeId, sumEmbNodeId);
            addVisEdge(edges, posEmbNodeId, sumEmbNodeId);
            lastEmbeddingNode = sumEmbNodeId;
        } else {
            // For RoPE/ALiBi, token embeddings are the main path forward from this block.
            // The posEmbNode is a standalone note.
            // No explicit sum node for RoPE/ALiBi at this stage.
        }
    }
    currentLevel++;

    const embLNNodeId = getNextVisNodeId();
    addVisNode(nodes, embLNNodeId, 'LayerNorm\n(Input)', { level: currentLevel, color: { background: '#BBDEFB', border: '#90CAF9' } });
    addVisEdge(edges, lastEmbeddingNode, embLNNodeId); // This was the line causing the error (303)
    let lastBlockOutputNodeId = embLNNodeId;
    currentLevel++;


    // --- 2. Transformer Layers ---
    let layerIdx = 0;

    while (layerIdx < numLayers) {
        const currentLayerConfig = layerConfigs[layerIdx];
        const currentSignature = getLayerSignature(
            currentLayerConfig, hiddenSize, numAttentionHeads, intermediateSizeDense, intermediateSizeMoe, moeSharedExperts
        );

        let consecutiveCount = 0;
        for (let tempIdx = layerIdx; tempIdx < numLayers; tempIdx++) {
            if (getLayerSignature(layerConfigs[tempIdx], hiddenSize, numAttentionHeads, intermediateSizeDense, intermediateSizeMoe, moeSharedExperts) === currentSignature) {
                consecutiveCount++;
            } else {
                break;
            }
        }

        const blockTitleNodeId = getNextVisNodeId();
        let blockLabel = ` Transformer Block (Layers ${layerIdx + 1}${consecutiveCount > 1 ? `-${layerIdx + consecutiveCount}` : ''})\nType: ${currentLayerConfig.type.toUpperCase()}${consecutiveCount > 1 ? ` (Repeats x${consecutiveCount})` : ''} `;
        addVisNode(nodes, blockTitleNodeId, blockLabel, { level: currentLevel, shape: 'hexagon', color: { background: '#D7DBDD', border: '#AEB6BF' }, font: { size: 12, bold: true } });
        addVisEdge(edges, lastBlockOutputNodeId, blockTitleNodeId, { length: 80 }); // Shorter edge
        let blockContentStartNode = blockTitleNodeId; // Residuals connect relative to this block's main flow input
        currentLevel++;

        const ln1Id = getNextVisNodeId();
        addVisNode(nodes, ln1Id, 'LayerNorm\n(Pre-MHA)', { level: currentLevel, color: { background: '#BBDEFB', border: '#90CAF9' } });
        addVisEdge(edges, blockContentStartNode, ln1Id);
        let mhaInputForResidual = ln1Id;

        const mhaResult = buildMhaBlock(nodes, edges, ln1Id, currentLevel + 1, hiddenSize, numAttentionHeads, positionalEmbeddingType);
        let mhaOutputNodeId = mhaResult.outputNodeId;
        currentLevel = mhaResult.nextLevel;

        const addNorm1Id = getNextVisNodeId();
        addVisNode(nodes, addNorm1Id, 'Add + LayerNorm\n(Post-MHA)', { level: currentLevel, color: { background: '#BBDEFB', border: '#90CAF9' } });
        addVisEdge(edges, mhaOutputNodeId, addNorm1Id);
        // Residual from input of MHA block (output of LN1) to input of AddNorm1
        addVisEdge(edges, mhaInputForResidual, addNorm1Id, {
            dashes: true, label: 'Res', color: { color: '#777777' },
            smooth: { type: 'curvedCW', roundness: 0.3 } // Attempt to arc residual
        });
        let ffnInputForResidual = addNorm1Id;
        currentLevel++;

        const ffnResult = buildFfnBlock(nodes, edges, addNorm1Id, currentLevel, hiddenSize, currentLayerConfig.type === 'dense' ? intermediateSizeDense : intermediateSizeMoe, currentLayerConfig, moeSharedExperts);
        let ffnOutputNodeId = ffnResult.outputNodeId;
        currentLevel = ffnResult.nextLevel;

        const addNorm2Id = getNextVisNodeId();
        addVisNode(nodes, addNorm2Id, 'Add + LayerNorm\n(Post-FFN)', { level: currentLevel, color: { background: '#BBDEFB', border: '#90CAF9' } });
        addVisEdge(edges, ffnOutputNodeId, addNorm2Id);
        // Residual from input of FFN block (output of AddNorm1) to input of AddNorm2
        addVisEdge(edges, ffnInputForResidual, addNorm2Id, {
            dashes: true, label: 'Res', color: { color: '#777777' },
            smooth: { type: 'curvedCW', roundness: 0.3 } // Attempt to arc residual
        });

        lastBlockOutputNodeId = addNorm2Id;
        currentLevel++;
        layerIdx += consecutiveCount;
    }

    // --- 3. Output Layer (if model has layers) ---
    if (numLayers > 0) {
        const finalLNId = getNextVisNodeId();
        addVisNode(nodes, finalLNId, 'LayerNorm\n(Final)', { level: currentLevel, color: { background: '#BBDEFB', border: '#90CAF9' } });
        addVisEdge(edges, lastBlockOutputNodeId, finalLNId);
        currentLevel++;

        const lmHeadId = getNextVisNodeId();
        addVisNode(nodes, lmHeadId, `Linear Head\n(${hiddenSize} → Vocab Size)`, { level: currentLevel, color: { background: '#E0E0E0', border: '#BDBDBD' } });
        addVisEdge(edges, finalLNId, lmHeadId);
        currentLevel++;

        const softmaxId = getNextVisNodeId();
        addVisNode(nodes, softmaxId, 'Softmax\n(Output)', { level: currentLevel, color: { background: '#E0E0E0', border: '#BDBDBD' } });
        addVisEdge(edges, lmHeadId, softmaxId);
    } else { // If no layers, connect input LN directly to a conceptual output if needed, or just end.
      // For now, if numLayers is 0, the graph will be very simple (just embeddings).
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
                enabled: true,
                direction: 'UD', // Up-Down
                sortMethod: 'directed',
                levelSeparation: 90,  // Reduced for tighter vertical packing
                nodeSpacing: 120,     // Space between nodes on same level (for horizontal MHA/MoE)
                treeSpacing: 150,     // Space between distinct trees (less relevant here)
                blockShifting: true,
                edgeMinimization: true,
                parentCentralization: false, // Experiment with this
            },
        },
        physics: {
            enabled: false, // Keep false for stable hierarchical layout
        },
        nodes: { // Default node styles (can be overridden per node)
            // shape properties defined in addVisNode
            font: { color: '#333333' }, // Default text color
             widthConstraint: { maximum: 150 }, // Prevent nodes from becoming too wide
        },
        edges: { // Default edge styles (can be overridden per edge)
            // smooth properties defined in addVisEdge or per edge
            width: 1,
            selectionWidth: 2,
        },
        interaction: {
            hover: true,
            tooltipDelay: 200,
            dragNodes: true,
            dragView: true,
            zoomView: true,
            navigationButtons: true, // Adds pan/zoom buttons
            keyboard: true, // Allows keyboard navigation
        },
    };

    const network = new vis.Network(container, data, options);

    // Reduce top empty space by focusing the view if possible, or by ensuring levels start near top.
    // The `level: 0` for initial nodes helps.
    // The dynamic height adjustment can also be fine-tuned.
    const finalLevel = (nodes.length > 0) ? Math.max(...nodes.map(n => n.level || 0)) : 0;
    const estimatedHeight = Math.max(parseInt(container.style.minHeight) || 650, (finalLevel + 1) * (options.layout.hierarchical.levelSeparation * 0.9) + 100);
    container.style.height = `${estimatedHeight}px`;

    // Try to fit the network after drawing
    network.once("stabilizationIterationsDone", function () {
        try { network.fit({ animation: { duration: 500, easingFunction: 'easeInOutQuad' } }); } catch(e) { /* ignore potential errors if view is too small */ }
    });
    // If physics is disabled, stabilizationIterationsDone might not fire reliably for fit.
    // A small timeout might be an alternative for fitting after render.
    setTimeout(() => {
         try { network.fit({animation: {duration: 300, easingFunction: 'linear'}}); } catch(e) { console.log("Fit error after timeout", e)}
    }, 200);
}

// Optional: Initial draw on load after a small delay for elements to be ready
window.addEventListener('load', () => {
    // Ensure default layer types match default layer number
    const numLayersInput = document.getElementById('numLayers');
    const layerTypesInput = document.getElementById('layerTypes');
    if (numLayersInput && layerTypesInput) {
        const numLayers = parseInt(numLayersInput.value);
        const types = layerTypesInput.value.split(',').map(s=>s.trim()).filter(s=>s);
        if (types.length !== numLayers && numLayers > 0) {
            // Create a default matching string
            let defaultTypes = [];
            for(let i=0; i<numLayers; i++) {
                defaultTypes.push(i % 2 === 0 ? 'dense' : 'moe:8');
            }
            layerTypesInput.value = defaultTypes.join(',');
        } else if (numLayers === 0) {
            layerTypesInput.value = "";
        }
    }

    // setTimeout(drawTransformer, 100); // Or trigger via button click only
});