
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
    const textMargin = 5;

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
              .append('rect')
              .classed(boundingBoxClass, true)
              .attr('x', textElement.attr('x') - textMargin)
              .attr('y', textElement.attr('y') - textElement.node().getBBox().height / 2 - 2 * textMargin)
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

    const defineArrowHead = (encompassingSvg) => {
	    // <defs>
	    //   <marker id="arrowhead" markerWidth="10" markerHeight="10" refY="3" orient="auto">
	    // 	<polygon points="0 0, 6 3, 0 6" />
	    //   </marker>
	    // </defs>
	const defs = encompassingSvg.append('defs');
	const marker = defs.append('marker')
	      .attr('markerWidth', '10')
	      .attr('markerHeight', '10')
	      .attr('refX', '6')
	      .attr('refY', '3')
	      .attr('orient', 'auto')
	      .attr('id', 'arrowhead');
        const polygon = marker.append('polygon')
	      .attr('points', '0 0, 6 3, 0 6');
    };
    
    
    const drawArrow = (encompassingSvg, [x1, y1], [x2, y2]) => {
        console.log(`x1 ${JSON.stringify(x1)}`);
        console.log(`y1 ${JSON.stringify(y1)}`);
        console.log(`x2 ${JSON.stringify(x2)}`);
        console.log(`y2 ${JSON.stringify(y2)}`);
        const line = encompassingSvg
              .append('line')
	      .attr('marker-end','url(#arrowhead)')
              .moveToBack() // not strictly necessary
	      .attr('x1', x1)
	      .attr('y1', y1)
	      .attr('x2', x2)
	      .attr('y2', y2)
              .classed('arrow-line', true);
    };
    
    /******************/
    /* Visualizations */
    /******************/
    
    { // RNN Depiction

        /* Init */
        
        const svg = d3.select('#rnn-depiction'); 
        svg
	    .attr('width', `${800}px`)
	    .attr('height', `${1200}px`);
        defineArrowHead(svg);
        const svgWidth = parseFloat(svg.style('width'));

        /* Blocks */
        
        // Words
        const words = ['"I"', '"loved"', '"it!"'];
        const wordDicts = words.reduce((accumulator, word, i) => {
            const textCenterX = xCenterPositionForIndex(svg, i, words.length);
            const wordGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', textCenterX, 100, word);
            accumulator.push({
                'centerX': textCenterX,
                'd3Handle': wordGroup,
            });
            return accumulator;
        }, []);

        // Embedding Layer
        const embeddingDicts = wordDicts.reduce((accumulator, wordDict) => {
            const centerX = wordDict.centerX;            
            const embeddingGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', centerX, 200, 'Embedding Layer');
            accumulator.push({
                'centerX': centerX,
                'd3Handle': embeddingGroup,
            });
            return accumulator;
        }, []);

        // LSTM Layer
        const LSTMDicts = wordDicts.reduce((accumulator, wordDict) => {
            const centerX = wordDict.centerX;
            const LSTMGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', centerX, 300, 'BiLSTM');
            accumulator.push({
                'centerX': centerX,
                'd3Handle': LSTMGroup,
            });
            return accumulator;
        }, []);

        // Attention Layer
        const attentionDicts = wordDicts.reduce((accumulator, wordDict) => {
            const centerX = wordDict.centerX;
            const attentionGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', centerX, 400, 'Self-Attention');
            accumulator.push({
                'centerX': centerX,
                'd3Handle': attentionGroup,
            });
            return accumulator;
        }, []);

        // Attention Softmax Layer
        const attentionSoftmaxCenterX = svgWidth/2;
        const attentionSoftmaxGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', attentionSoftmaxCenterX, 500, 'Softmax');
        const attentionSoftmaxDict = {
            'centerX': attentionSoftmaxCenterX,
            'd3Handle': attentionSoftmaxGroup,
        };

        // Attention Sum Layer
        const attentionSumCenterX = svgWidth/2;
        const attentionSumGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', attentionSumCenterX, 600, '+');
        const attentionSumDict = {
            'centerX': attentionSumCenterX,
            'd3Handle': attentionSumGroup,
        };
        
        // Fully Connected Layer
        const fullyConnectedCenterX = svgWidth/2;
        const fullyConnectedGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', fullyConnectedCenterX, 700, 'Fully Connected Layer');
        const fullyConnectedDict = {
            'centerX': fullyConnectedCenterX,
            'd3Handle': fullyConnectedGroup,
        };

        // Sigmoid Layer
        const SigmoidDicts = wordDicts.reduce((accumulator, wordDict) => {
            const centerX = wordDict.centerX;
            const SigmoidGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', centerX, 800, 'Sigmoid');
            accumulator.push({
                'centerX': centerX,
                'd3Handle': SigmoidGroup,
            });
            return accumulator;
        }, []);
        
        // Output Layer
        const outputDicts = wordDicts.reduce((accumulator, wordDict, i) => {
            const centerX = wordDict.centerX;
            const LSTMGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', centerX, 900, `Class ${i}`);
            accumulator.push({
                'centerX': centerX,
                'd3Handle': LSTMGroup,
            });
            return accumulator;
        }, []);

        /* Arrows */

        // Words to Embedding Layer
        zip([wordDicts, embeddingDicts]).forEach(([wordDict, embeddingDict]) => {
            drawArrow(svg, getD3HandleBottomXY(wordDict.d3Handle), getD3HandleTopXY(embeddingDict.d3Handle));
        });
        
        // Embedding Layer to LSTM Layer
        zip([embeddingDicts, LSTMDicts]).forEach(([embeddingDict, LSTMDict]) => {
            drawArrow(svg, getD3HandleBottomXY(embeddingDict.d3Handle), getD3HandleTopXY(LSTMDict.d3Handle));
        });
        
        // LSTM Layer to Attention Layer
        zip([LSTMDicts, attentionDicts]).forEach(([LSTMDict, attentionDict]) => {
            drawArrow(svg, getD3HandleBottomXY(LSTMDict.d3Handle), getD3HandleTopXY(attentionDict.d3Handle));
        });
        
        // Attention Layer to Softmax
        attentionDicts.forEach((attentionDict) => {
            drawArrow(svg, getD3HandleBottomXY(attentionDict.d3Handle), getD3HandleTopXY(attentionSoftmaxGroup));
        });
        
    };

    { // CNN Depiction
        
    };

    { // DNN Depiction
        
    };
    
};
