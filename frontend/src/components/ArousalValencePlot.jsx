import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

export default function ArousalValencePlot({ arousal = 0.5, valence = 0.5, history = [] }) {
  const svgRef = useRef(null);

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const W = 260, H = 220;
    const margin = { top: 10, right: 10, bottom: 30, left: 30 };
    const w = W - margin.left - margin.right;
    const h = H - margin.top - margin.bottom;

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const xScale = d3.scaleLinear().domain([0, 1]).range([0, w]);
    const yScale = d3.scaleLinear().domain([0, 1]).range([h, 0]);

    // Quadrant backgrounds
    const quadrants = [
      { x: 0, y: 0.5, w: 0.5, h: 0.5, color: '#58a6ff', label: 'Underload' },
      { x: 0.5, y: 0.5, w: 0.5, h: 0.5, color: '#3fb950', label: 'Optimal' },
      { x: 0, y: 0, w: 0.5, h: 0.5, color: '#d29922', label: 'Fatigue' },
      { x: 0.5, y: 0, w: 0.5, h: 0.5, color: '#f78166', label: 'Overload' },
    ];
    quadrants.forEach(q => {
      g.append('rect')
        .attr('x', xScale(q.x)).attr('y', yScale(q.y + q.h))
        .attr('width', xScale(q.w)).attr('height', yScale(0) - yScale(q.h))
        .attr('fill', q.color).attr('opacity', 0.08);
      g.append('text')
        .attr('x', xScale(q.x + q.w / 2)).attr('y', yScale(q.y + q.h / 2))
        .attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')
        .attr('fill', q.color).attr('font-size', '10px').attr('opacity', 0.6)
        .text(q.label);
    });

    // Axes
    g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(xScale).ticks(5))
      .selectAll('text, line, path').attr('stroke', '#30363d').attr('fill', '#8b949e');
    g.append('g').call(d3.axisLeft(yScale).ticks(5))
      .selectAll('text, line, path').attr('stroke', '#30363d').attr('fill', '#8b949e');

    // Axis labels
    g.append('text').attr('x', w / 2).attr('y', h + 26)
      .attr('text-anchor', 'middle').attr('fill', '#8b949e').attr('font-size', '11px').text('Valence →');
    g.append('text').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -22)
      .attr('text-anchor', 'middle').attr('fill', '#8b949e').attr('font-size', '11px').text('Arousal →');

    // History trail
    if (history.length > 1) {
      const line = d3.line()
        .x(d => xScale(d.valence ?? 0.5))
        .y(d => yScale(d.arousal ?? 0.5))
        .curve(d3.curveCatmullRom);
      g.append('path')
        .datum(history)
        .attr('d', line)
        .attr('fill', 'none')
        .attr('stroke', '#58a6ff')
        .attr('stroke-width', 1.5)
        .attr('opacity', 0.4);
    }

    // Current point
    g.append('circle')
      .attr('cx', xScale(valence)).attr('cy', yScale(arousal))
      .attr('r', 8).attr('fill', '#58a6ff').attr('stroke', '#e6edf3').attr('stroke-width', 2);

  }, [arousal, valence, history]);

  return <svg ref={svgRef} width="260" height="220" style={{ display: 'block', margin: '0 auto' }} />;
}
