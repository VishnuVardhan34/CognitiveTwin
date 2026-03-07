import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

const CLASS_COLORS = ['#58a6ff', '#3fb950', '#f78166', '#d29922'];
const CLASS_NAMES  = ['Underload', 'Optimal', 'Overload', 'Fatigue'];
const ARC_START    = -Math.PI * 0.75;
const ARC_END      =  Math.PI * 0.75;

export default function CognitiveGauge({ value = 1, confidence = 0.5, classProbs = {} }) {
  const svgRef = useRef(null);

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const W = 220, H = 160;
    const cx = W / 2, cy = H * 0.85;
    const outerR = 80, innerR = 55;

    const g = svg.append('g');

    // Background arc segments (4 classes)
    const arcStep = (ARC_END - ARC_START) / 4;
    CLASS_NAMES.forEach((name, i) => {
      const arc = d3.arc()
        .innerRadius(innerR)
        .outerRadius(outerR)
        .startAngle(ARC_START + i * arcStep)
        .endAngle(ARC_START + (i + 1) * arcStep);
      g.append('path')
        .attr('d', arc)
        .attr('fill', CLASS_COLORS[i])
        .attr('opacity', 0.25)
        .attr('transform', `translate(${cx},${cy})`);
    });

    // Active arc segment
    const arcActive = d3.arc()
      .innerRadius(innerR)
      .outerRadius(outerR)
      .startAngle(ARC_START + value * arcStep)
      .endAngle(ARC_START + (value + 1) * arcStep);
    g.append('path')
      .attr('d', arcActive)
      .attr('fill', CLASS_COLORS[value])
      .attr('opacity', 0.9)
      .attr('transform', `translate(${cx},${cy})`);

    // Needle
    const angle = ARC_START + (value + 0.5) * arcStep;
    const needleLen = outerR - 8;
    const nx = cx + needleLen * Math.sin(angle);
    const ny = cy - needleLen * Math.cos(angle);
    g.append('line')
      .attr('x1', cx).attr('y1', cy)
      .attr('x2', nx).attr('y2', ny)
      .attr('stroke', '#e6edf3')
      .attr('stroke-width', 2.5)
      .attr('stroke-linecap', 'round');
    g.append('circle')
      .attr('cx', cx).attr('cy', cy)
      .attr('r', 5)
      .attr('fill', '#e6edf3');

    // Centre label
    g.append('text')
      .attr('x', cx).attr('y', cy - 10)
      .attr('text-anchor', 'middle')
      .attr('fill', CLASS_COLORS[value])
      .attr('font-size', '14px')
      .attr('font-weight', '700')
      .text(CLASS_NAMES[value]);

    g.append('text')
      .attr('x', cx).attr('y', cy + 8)
      .attr('text-anchor', 'middle')
      .attr('fill', '#8b949e')
      .attr('font-size', '11px')
      .text(`${(confidence * 100).toFixed(0)}% conf`);

    // Class probability bars below gauge
    const barY0 = cy + 24;
    CLASS_NAMES.forEach((name, i) => {
      const prob = classProbs[name] || 0;
      const bx = cx - 80 + i * 40;
      g.append('rect')
        .attr('x', bx).attr('y', barY0)
        .attr('width', 32).attr('height', 4)
        .attr('rx', 2)
        .attr('fill', CLASS_COLORS[i])
        .attr('opacity', 0.3);
      g.append('rect')
        .attr('x', bx).attr('y', barY0)
        .attr('width', 32 * prob).attr('height', 4)
        .attr('rx', 2)
        .attr('fill', CLASS_COLORS[i])
        .attr('opacity', 0.9);
      g.append('text')
        .attr('x', bx + 16).attr('y', barY0 + 16)
        .attr('text-anchor', 'middle')
        .attr('fill', '#8b949e')
        .attr('font-size', '9px')
        .text(name.slice(0, 3));
    });

  }, [value, confidence, classProbs]);

  return <svg ref={svgRef} width="220" height="175" style={{ display: 'block', margin: '0 auto' }} />;
}
