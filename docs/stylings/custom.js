
!(function (e) {
    "use strict";
    e('a.js-scroll-trigger[href*="#"]:not([href="#"])').click(function () {
        if (location.pathname.replace(/^\//, "") == this.pathname.replace(/^\//, "") && location.hostname == this.hostname) {
            let t = e(this.hash);
            if ((t = t.length ? t : e("[name=" + this.hash.slice(1) + "]")).length) {
                return e("html, body").animate({ scrollTop: t.offset().top - 60 }, 1e3, "easeInOutExpo"), !1;
            }
        }
        return null;
    }),
        e(".js-scroll-trigger").click(function () {
            e(".navbar-collapse").collapse("hide");
        }),
        e("body").scrollspy({ target: "#navigator" });
})(jQuery);

{ // Architecture Depictions

    /*******************/
    /* Misc. Utilities */
    /*******************/
    
    const isUndefined = value => value === void(0);

    const shuffle = (array) => array.sort(() => Math.random() - 0.5);

    const zip = rows => rows[0].map((_,c) => rows.map(row => row[c]));
    
    const createNewElement = (childTag, {classes, attributes, innerHTML}={}) => {
        const newElement = childTag === 'svg' ? document.createElementNS('http://www.w3.org/2000/svg', childTag) : document.createElement(childTag);
        if (!isUndefined(classes)) {
            classes.forEach(childClass => newElement.classList.add(childClass));
        }
        if (!isUndefined(attributes)) {
            Object.entries(attributes).forEach(([attributeName, attributeValue]) => {
                newElement.setAttribute(attributeName, attributeValue);
            });
        }
        if (!isUndefined(innerHTML)) {
            newElement.innerHTML = innerHTML;
        }
        return newElement;
    };

    // D3 Extensions
    d3.selection.prototype.moveToFront = function() {
	return this.each(function() {
	    if (this.parentNode !== null) {
		this.parentNode.appendChild(this);
	    }
	});
    };

    d3.selection.prototype.moveToBack = function() {
        return this.each(function() {
            var firstChild = this.parentNode.firstChild;
            if (firstChild) {
                this.parentNode.insertBefore(this, firstChild);
            }
        });
    };

    /***************************/
    /* Visualization Utilities */
    /***************************/
    
    const innerMargin = 150;
    const textMargin = 8;
    const curvedArrowOffset = 30;

    const xCenterPositionForIndex = (encompassingSvg, index, total) => {
        const svgWidth = parseFloat(encompassingSvg.style('width'));
        const innerWidth = svgWidth - 2 * innerMargin;
        const delta = innerWidth / (total - 1);
        const centerX = innerMargin + index * delta;
        return centerX;
    };

    const generateTextWithBoundingBox = (encompassingSvg, parentGroupClass, textElementClass, boundingBoxClass, textCenterX, yPosition, textString) => {
        const parentGroup = encompassingSvg
              .append('g')
              .classed(parentGroupClass, true);
        const textElement = parentGroup
              .append('text')
	      .attr('y', yPosition)
              .classed(textElementClass, true)
              .html(textString);
        textElement
	    .attr('x', textCenterX - textElement.node().getBBox().width / 2);
        const boundingBoxElement = parentGroup
              .append('rect');
        boundingBoxElement
              .classed(boundingBoxClass, true)
              .attr('x', textElement.attr('x') - textMargin)
              .attr('y', () => textElement.node().getBBox().y - textMargin)
              .attr('width', textElement.node().getBBox().width + 2 * textMargin)
              .attr('height', textElement.node().getBBox().height + 2 * textMargin);
        textElement.moveToFront();
        return parentGroup;
    };

    const getD3HandleTopXY = (element) => {
        /* element is a D3 handle */
        const boundingBox = element.node().getBBox();
        const x = boundingBox.x + boundingBox.width/2;
        const y = boundingBox.y;
        return [x, y];
    };

    const getD3HandleBottomXY = (element) => {
        /* element is a D3 handle */
        const boundingBox = element.node().getBBox();
        const x = boundingBox.x + boundingBox.width/2;
        const y = boundingBox.y + boundingBox.height;
        return [x, y];
    };

    const getD3HandleTopLeftXY = (element) => {
        /* element is a D3 handle */
        const boundingBox = element.node().getBBox();
        const x = boundingBox.x;
        const y = boundingBox.y;
        return [x, y];
    };

    const getD3HandleBottomLeftXY = (element) => {
        /* element is a D3 handle */
        const boundingBox = element.node().getBBox();
        const x = boundingBox.x;
        const y = boundingBox.y + boundingBox.height;
        return [x, y];
    };

    const getD3HandleTopRightXY = (element) => {
        /* element is a D3 handle */
        const boundingBox = element.node().getBBox();
        const x = boundingBox.x + boundingBox.width;
        const y = boundingBox.y;
        return [x, y];
    };

    const getD3HandleBottomRightXY = (element) => {
        /* element is a D3 handle */
        const boundingBox = element.node().getBBox();
        const x = boundingBox.x + boundingBox.width;
        const y = boundingBox.y + boundingBox.height;
        return [x, y];
    };

    const defineArrowHead = (encompassingSvg) => {
	const defs = encompassingSvg.append('defs');
	const marker = defs.append('marker')
	      .attr('markerWidth', '10')
	      .attr('markerHeight', '10')
	      .attr('refX', '5')
	      .attr('refY', '3')
	      .attr('orient', 'auto')
	      .attr('id', 'arrowhead');
        const polygon = marker.append('polygon')
	      .attr('points', '0 0, 6 3, 0 6');
    };
    
    
    const drawArrow = (encompassingSvg, [x1, y1], [x2, y2]) => {
        const line = encompassingSvg
              .append('line')
	      .attr('marker-end','url(#arrowhead)')
              .moveToBack()
	      .attr('x1', x1)
	      .attr('y1', y1)
	      .attr('x2', x2)
	      .attr('y2', y2)
              .classed('arrow-line', true);
    };
    
    const drawCurvedArrow = (encompassingSvg, [x1, y1], [x2, y2]) => {
	const midpointX = (x1+x2)/2;
	const midpointY = (y1+y2)/2;
	const dx = (x2 - x1);
	const dy = (y2 - y1);
	const normalization = Math.sqrt((dx * dx) + (dy * dy));
	const offSetX = midpointX + curvedArrowOffset*(dy/normalization);
	const offSetY = midpointY - curvedArrowOffset*(dx/normalization);
	const path = `M ${x1}, ${y1} S ${offSetX}, ${offSetY} ${x2}, ${y2}`;
        const line = encompassingSvg
              .append('path')
	      .attr('marker-end','url(#arrowhead)')
              .moveToBack()
	      .attr('d', path)
              .classed('arrow-line', true);
    };
    
    /******************/
    /* Visualizations */
    /******************/
    
    const renderRNNArchitecture = () => {

        /* Init */
        
        const svg = d3.select('#rnn-depiction');
        svg.selectAll('*').remove();
        svg
	    .attr('width', `80vw`)
	    .attr('height', `${1000}px`);
        defineArrowHead(svg);
        const svgWidth = parseFloat(svg.style('width'));

        /* Blocks */

        const words = ['"The"', '"oil"', '"prices"', '&hellip;', '"significantly."'];
        const outputClassCount = 4+Math.floor(Math.random()*3);
        
        // Words
        const wordGroups = words.map((word, i) => {
            const textCenterX = xCenterPositionForIndex(svg, i, words.length);
            const wordGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', textCenterX, 100, word);
            wordGroup.classed('word-group', true);
            return wordGroup;
        });

        // Embedding Layer
        const embeddingGroups = wordGroups.map(wordGroup => {
            const centerX = getD3HandleTopXY(wordGroup)[0];
            const embeddingGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', centerX, 200, 'Embedding Layer');
            embeddingGroup.classed('embedding-group', true);
            return embeddingGroup;
        });

        // LSTM Layer
        const LSTMGroups = wordGroups.map(wordGroup => {
            const centerX = getD3HandleTopXY(wordGroup)[0];
            const LSTMGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', centerX, 300, 'BiLSTM');
            LSTMGroup.classed('lstm-group', true);
            return LSTMGroup;
        });

        // Attention Layer
        const attentionGroups = wordGroups.map(wordGroup => {
            const centerX = getD3HandleTopXY(wordGroup)[0];
            const attentionGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', centerX, 400, 'Attention');
            attentionGroup.classed('attention-group', true);
            return attentionGroup;
        });

        // Attention Softmax Layer
        const attentionSoftmaxCenterX = svgWidth/2;
        const attentionSoftmaxGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', attentionSoftmaxCenterX, 500, 'Softmax');
        attentionSoftmaxGroup.classed('attention-softmax-group', true);
        const attentionSoftmaxGroupLeftX = attentionGroups[0].node().getBBox().x;
        const rightmostAttentionGroupBoundingBox = attentionGroups[words.length-1].node().getBBox();
        const attentionSoftmaxGroupRightX = rightmostAttentionGroupBoundingBox.x + rightmostAttentionGroupBoundingBox.width;
        attentionSoftmaxGroup.select('rect')
            .attr('x', attentionSoftmaxGroupLeftX)
            .attr('width', attentionSoftmaxGroupRightX - attentionSoftmaxGroupLeftX);

        // Attention Sum Layer
        const attentionSumCenterX = svgWidth/2;
        const attentionSumGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', attentionSumCenterX, 600, '+');
        attentionSumGroup.classed('attention-sum-group', true);
        
4        // Fully Connected Layer
        const fullyConnectedCenterX = svgWidth/2;
        const fullyConnectedGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', fullyConnectedCenterX, 700, 'Fully Connected Layer');
        fullyConnectedGroup.classed('fully-connected-group', true);
        
        // Sigmoid Layer
        const sigmoidGroups = [];
        for(let i=0; i<outputClassCount; i++) {
            const centerX = xCenterPositionForIndex(svg, i, outputClassCount);
            const sigmoidGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', centerX, 800, 'Sigmoid');
            sigmoidGroup.classed('sigmoid-group', true);
            sigmoidGroups.push(sigmoidGroup);
        };
        
        // Output Layer
        const outputGroups = sigmoidGroups.map((sigmoidGroup, i) => {
            const centerX = getD3HandleTopXY(sigmoidGroup)[0];
            const outputGroupLabelText = i === outputClassCount-1 ? `Label n Score: ${Math.random().toFixed(4)}` : (i === outputClassCount-2) ? '&hellip;' : `Label ${i} Score: ${Math.random().toFixed(4)}`;
            const outputGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', centerX, 900, outputGroupLabelText);
            outputGroup.classed('output-group', true);
            return outputGroup;
        });

        /* Arrows */

        // Words to Embedding Layer
        zip([wordGroups, embeddingGroups]).forEach(([wordGroup, embeddingGroup]) => {
            drawArrow(svg, getD3HandleBottomXY(wordGroup), getD3HandleTopXY(embeddingGroup));
        });
        
        // Embedding Layer to LSTM Layer
        zip([embeddingGroups, LSTMGroups]).forEach(([embeddingGroup, LSTMGroup]) => {
            drawArrow(svg, getD3HandleBottomXY(embeddingGroup), getD3HandleTopXY(LSTMGroup));
        });

        // Intra-LSTM Layer Arrows
        LSTMGroups.forEach((LSTMGroup, i) => {
            if (i<LSTMGroups.length-1) {
                const nextLSTMGroup = LSTMGroups[i+1];
                drawCurvedArrow(svg, getD3HandleTopRightXY(LSTMGroup), getD3HandleTopLeftXY(nextLSTMGroup));
                drawCurvedArrow(svg, getD3HandleBottomLeftXY(nextLSTMGroup), getD3HandleBottomRightXY(LSTMGroup));
            }
	});
        
        // LSTM Layer to Attention Layer
        zip([LSTMGroups, attentionGroups]).forEach(([LSTMGroup, attentionGroup]) => {
            drawArrow(svg, getD3HandleBottomXY(LSTMGroup), getD3HandleTopXY(attentionGroup));
        });
        
        // Attention Layer to Softmax Layer
        attentionGroups.forEach((attentionGroup) => {
            const [attentionGroupBottomX, attentionGroupBottomY] = getD3HandleBottomXY(attentionGroup);
            const attentionSoftmaxGroupTopY = getD3HandleTopXY(attentionSoftmaxGroup)[1];
            drawArrow(svg, [attentionGroupBottomX, attentionGroupBottomY], [attentionGroupBottomX, attentionSoftmaxGroupTopY]);
        });
        
        // Softmax Layer to Attention Sum Layer
        attentionGroups.forEach((attentionGroup) => {
            const attentionGroupBottomX = getD3HandleBottomXY(attentionGroup)[0];
            const attentionSoftmaxGroupBottomY = getD3HandleBottomXY(attentionSoftmaxGroup)[1];
            drawArrow(svg, [attentionGroupBottomX, attentionSoftmaxGroupBottomY], getD3HandleTopXY(attentionSumGroup));
        });

        // Attention Sum Layer to Fully Connected Layer
        drawArrow(svg, getD3HandleBottomXY(attentionSumGroup), getD3HandleTopXY(fullyConnectedGroup));

        // Fully Connected Layer to Sigmoid Layer
        sigmoidGroups.forEach((sigmoidGroup) => {
            drawArrow(svg, getD3HandleBottomXY(fullyConnectedGroup), getD3HandleTopXY(sigmoidGroup));
        });

        // Sigmoid Layer to Output Layer
        zip([sigmoidGroups, outputGroups]).forEach(([sigmoidGroup, outputGroup]) => {
            drawArrow(svg, getD3HandleBottomXY(sigmoidGroup), getD3HandleTopXY(outputGroup));
        });
        
    };
    setTimeout(renderRNNArchitecture, 1000); // @todo this is a workaround
    window.addEventListener('resize', renderRNNArchitecture);

    const renderCNNArchitecture = () => {

        /* Init */
        
        const svg = d3.select('#cnn-depiction');
        svg.selectAll('*').remove();
        svg
	    .attr('width', `80vw`)
	    .attr('height', `${1000}px`);
        defineArrowHead(svg);
        const svgWidth = parseFloat(svg.style('width'));

        /* Blocks */

        const words = ['"Grain"', '"shipments"', '"and"', '&hellip;', '"possible."'];
        const convolutionLayerCount = words.length-1+Math.floor(Math.random()*3);
        const convolutionSizeVariableNames = 'XYZWPQRSTUV';
        const convolutionColors = shuffle([
            '#ffc4c4',
            '#deffc4',
            '#c4ffdb',
            '#c4fffe',
            '#c4e0ff',
            '#cbc4ff',
            '#ecc4ff',
            '#ffc4e8',
        ]);
        const outputClassCount = 4+Math.floor(Math.random()*3);
        
        // Words
        const wordGroups = words.map((word, i) => {
            const textCenterX = xCenterPositionForIndex(svg, i, words.length);
            const wordGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', textCenterX, 100, word);
            wordGroup.classed('word-group', true);
            return wordGroup;
        });

        // Embedding Layer
        const embeddingGroups = wordGroups.map(wordGroup => {
            const centerX = getD3HandleTopXY(wordGroup)[0];
            const embeddingGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', centerX, 200, 'Embedding Layer');
            embeddingGroup.classed('embedding-group', true);
            return embeddingGroup;
        });

        // Convolution Layers
        const convolutionGroups = [];
        for (let i=0; i<convolutionLayerCount; i++) {
            const centerX = xCenterPositionForIndex(svg, i, convolutionLayerCount);
            const convolutionGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', centerX, 400, `Size-${convolutionSizeVariableNames[i]} Conv`);
            convolutionGroup.classed('convolution-group', true);
            convolutionGroup
                .select('.text-with-bbox-group-bounding-box')
                .style('fill', convolutionColors[i]);
            convolutionGroups.push(convolutionGroup);
        };

        // Pooling Layers
        const poolingGroups = convolutionGroups.map(convolutionGroup => {
            const centerX = getD3HandleTopXY(convolutionGroup)[0];
            const poolingGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', centerX, 500, 'Pool');
            poolingGroup.classed('pooling-group', true);
            return poolingGroup;
        });
        
        // Concatenation Layer
        const concatenationCenterX = svgWidth/2;
        const concatenationGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', concatenationCenterX, 600, 'Concatenation');
        concatenationGroup.classed('concatenation-group', true);
        
        // Prediction Layer
        const predictionCenterX = svgWidth/2;
        const predictionGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', predictionCenterX, 700, 'Prediction Layer');
        predictionGroup.classed('prediction-group', true);
        
        // Sigmoid Layer
        const sigmoidGroups = [];
        for(let i=0; i<outputClassCount; i++) {
            const centerX = xCenterPositionForIndex(svg, i, outputClassCount);
            const sigmoidGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', centerX, 800, 'Sigmoid');
            sigmoidGroup.classed('sigmoid-group', true);
            sigmoidGroups.push(sigmoidGroup);
        };
        
        // Output Layer
        const outputGroups = sigmoidGroups.map((sigmoidGroup, i) => {
            const centerX = getD3HandleTopXY(sigmoidGroup)[0];
            const outputGroupLabelText = i === outputClassCount-1 ? `Label n Score: ${Math.random().toFixed(4)}` : (i === outputClassCount-2) ? '&hellip;' : `Label ${i} Score: ${Math.random().toFixed(4)}`;
            const outputGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', centerX, 900, outputGroupLabelText);
            outputGroup.classed('output-group', true);
            return outputGroup;
        });

        /* Arrows */

        // Words to Embedding Layer
        zip([wordGroups, embeddingGroups]).forEach(([wordGroup, embeddingGroup]) => {
            drawArrow(svg, getD3HandleBottomXY(wordGroup), getD3HandleTopXY(embeddingGroup));
        });
        
        // Embedding Layer to Convolution Layers
        embeddingGroups.forEach(embeddingGroup => {
            convolutionGroups.forEach(convolutionGroup => {
                drawArrow(svg, getD3HandleBottomXY(embeddingGroup), getD3HandleTopXY(convolutionGroup));
            });
        });

        // Convolution Layers to Pooling Layer
        zip([convolutionGroups, poolingGroups]).forEach(([convolutionGroup, poolingGroup]) => {
            drawArrow(svg, getD3HandleBottomXY(convolutionGroup), getD3HandleTopXY(poolingGroup));
        });
        
        // Pooling Layer to Concatenation Layer
        poolingGroups.forEach(poolingGroup => {
            drawArrow(svg, getD3HandleBottomXY(poolingGroup), getD3HandleTopXY(concatenationGroup));
        });

        // Concatenation Layer to Prediction Layer
        drawArrow(svg, getD3HandleBottomXY(concatenationGroup), getD3HandleTopXY(predictionGroup));
        
        // Prediction Layer to Sigmoid Layer
        sigmoidGroups.forEach((sigmoidGroup) => {
            drawArrow(svg, getD3HandleBottomXY(predictionGroup), getD3HandleTopXY(sigmoidGroup));
        });

        // Sigmoid Layer to Output Layer
        zip([sigmoidGroups, outputGroups]).forEach(([sigmoidGroup, outputGroup]) => {
            drawArrow(svg, getD3HandleBottomXY(sigmoidGroup), getD3HandleTopXY(outputGroup));
        });
        
    };
    setTimeout(renderCNNArchitecture, 1000); // @todo this is a workaround
    window.addEventListener('resize', renderCNNArchitecture);
    
    const renderDNNArchitecture = () => {

        /* Init */
        
        const denseLayerCount = 3+Math.floor(Math.random()*3);
        const svg = d3.select('#dnn-depiction');
        svg.selectAll('*').remove();
        svg
	    .attr('width', `80vw`)
	    .attr('height', `${600+denseLayerCount*100}px`);
        defineArrowHead(svg);
        const svgWidth = parseFloat(svg.style('width'));

        /* Blocks */

        const words = ['"French"', '"prime"', '&hellip;', '"unsure."'];
        const outputClassCount = 4+Math.floor(Math.random()*3);
        
        // Words
        const wordGroups = words.map((word, i) => {
            const textCenterX = xCenterPositionForIndex(svg, i, words.length);
            const wordGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', textCenterX, 100, word);
            wordGroup.classed('word-group', true);
            return wordGroup;
        });

        // Embedding Layer
        const embeddingGroups = wordGroups.map(wordGroup => {
            const centerX = getD3HandleTopXY(wordGroup)[0];
            const embeddingGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', centerX, 200, 'Embedding Layer');
            embeddingGroup.classed('embedding-group', true);
            return embeddingGroup;
        });
        
        // Dense Layer
        const denseGroups = [];
        for(let i=0; i<denseLayerCount; i++) {
            const denseCenterX = svgWidth/2;
            const denseGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', denseCenterX, 300+i*100, i===denseLayerCount-2 ? '&hellip;' : 'Dense Layer');
            denseGroup.classed('dense-group', true);
            const denseGroupLeftX = embeddingGroups[0].node().getBBox().x;
            const rightmostAttentionGroupBoundingBox = embeddingGroups[words.length-1].node().getBBox();
            const denseGroupRightX = rightmostAttentionGroupBoundingBox.x + rightmostAttentionGroupBoundingBox.width;
            denseGroup.select('rect')
                .attr('x', denseGroupLeftX)
                .attr('width', denseGroupRightX - denseGroupLeftX);
            denseGroups.push(denseGroup);
        }

        // Fully Connected Layer
        const fullyConnectedCenterX = svgWidth/2;
        const fullyConnectedGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', fullyConnectedCenterX, 300+denseLayerCount*100, 'Fully Connected Layer');
        fullyConnectedGroup.classed('fully-connected-group', true);

        // Sigmoid Layer
        const sigmoidGroups = [];
        for(let i=0; i<outputClassCount; i++) {
            const centerX = xCenterPositionForIndex(svg, i, outputClassCount);
            const sigmoidGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', centerX, 400+denseLayerCount*100, 'Sigmoid');
            sigmoidGroup.classed('sigmoid-group', true);
            sigmoidGroups.push(sigmoidGroup);
        };
                
        // Output Layer
        const outputGroups = sigmoidGroups.map((sigmoidGroup, i) => {
            const centerX = getD3HandleTopXY(sigmoidGroup)[0];
            const outputGroupLabelText = i === outputClassCount-1 ? `Label n Score: ${Math.random().toFixed(4)}` : (i === outputClassCount-2) ? '&hellip;' : `Label ${i} Score: ${Math.random().toFixed(4)}`;
            const outputGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', centerX, 500+denseLayerCount*100, outputGroupLabelText);
            outputGroup.classed('output-group', true);
            return outputGroup;
        });

        /* Arrows */

        // Words to Embedding Layer
        zip([wordGroups, embeddingGroups]).forEach(([wordGroup, embeddingGroup]) => {
            drawArrow(svg, getD3HandleBottomXY(wordGroup), getD3HandleTopXY(embeddingGroup));
        });

        // Embedding Layer to First Dense Layer
        const denseGroupTopY = getD3HandleTopXY(denseGroups[0])[1];
        embeddingGroups.forEach(embeddingGroup0 => {
            const [embeddingGroup0BottomX, embeddingGroup0BottomY] = getD3HandleBottomXY(embeddingGroup0);
            embeddingGroups.forEach(embeddingGroup1 => {
                const embeddingGroup1BottomX = getD3HandleBottomXY(embeddingGroup1)[0];
                drawArrow(svg, [embeddingGroup0BottomX, embeddingGroup0BottomY], [embeddingGroup1BottomX, denseGroupTopY]);
            });
        });

        // Intra Dense Layer Arrows
        denseGroups.forEach((denseGroup, i) => {
            if (i < denseGroups.length-1) {
                const denseGroupY = getD3HandleBottomXY(denseGroup)[1];
                const nextDenseGroup = denseGroups[i+1];
                const nextDenseGroupY = getD3HandleTopXY(nextDenseGroup)[1];
                embeddingGroups.forEach(embeddingGroup0 => {
                    const embeddingGroup0X = getD3HandleBottomXY(embeddingGroup0)[0];
                    embeddingGroups.forEach(embeddingGroup1 => {
                        const embeddingGroup1X = getD3HandleBottomXY(embeddingGroup1)[0];
                        drawArrow(svg, [embeddingGroup0X, denseGroupY], [embeddingGroup1X, nextDenseGroupY]);
                    });
                });
            }
        });

        // Last Dense Layer to Fully Connected Layer
        embeddingGroups.forEach(embeddingGroup => {
            const embeddingGroupX = getD3HandleBottomXY(embeddingGroup)[0];
            const denseGroupY = getD3HandleBottomXY(denseGroups[denseGroups.length-1])[1];
            drawArrow(svg, [embeddingGroupX, denseGroupY], getD3HandleTopXY(fullyConnectedGroup));
        });
        
        // Fully Connected Layer to Sigmoid Layer
        sigmoidGroups.forEach((sigmoidGroup) => {
            drawArrow(svg, getD3HandleBottomXY(fullyConnectedGroup), getD3HandleTopXY(sigmoidGroup));
        });

        // Sigmoid Layer to Output Layer
        zip([sigmoidGroups, outputGroups]).forEach(([sigmoidGroup, outputGroup]) => {
            drawArrow(svg, getD3HandleBottomXY(sigmoidGroup), getD3HandleTopXY(outputGroup));
        });

    };
    setTimeout(renderDNNArchitecture, 1000); // @todo this is a workaround
    window.addEventListener('resize', renderDNNArchitecture);
    
};
