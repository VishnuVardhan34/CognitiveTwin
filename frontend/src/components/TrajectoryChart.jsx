import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

const CLASS_COLORS = ['#58a6ff', '#3fb950', '#f78166', '#d29922'];
const CLASS_NAMES  = ['Underload', 'Optimal', 'Overload', 'Fatigue'];

export default function TrajectoryChart({ trajectory = [] }) {
  const svgRef = useRef(null);

  useEffect(() => {
    const container = svgRef.current?.parentElement;
    const W = container ? container.clientWidth - 32 : 500;
    const H = 140;
    const margin = { top: 10, right: 10, bottom: 24, left: 36 };
    const w = W - margin.left - margin.right;
    const h = H - margin.top - margin.bottom;

    const svg = d3.select(svgRef.current)
      .attr('width', W).attr('height', H);
    svg.selectAll('*').remove();

    if (trajectory.length < 2) {
      svg.append('text').attr('x', W / 2).attr('y', H / 2)
        .attr('text-anchor', 'middle').attr('fill', '#8b949e')
        .attr('font-size', '12px').text('Waiting for data…');
      return;
    }

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const xScale = d3.scaleLinear()
      .domain([0, trajectory.length - 1])
      .range([0, w]);
    const yScale = d3.scaleLinear().domain([-0.5, 3.5]).range([h, 0]);

    // Horizontal class bands
    [0, 1, 2, 3].forEach(c => {
      g.append('line')
        .attr('x1', 0).attr('x2', w)
        .attr('y1', yScale(c)).attr('y2', yScale(c))
        .attr('stroke', CLASS_COLORS[c]).attr('stroke-width', 1)
        .attr('stroke-dasharray', '4,4').attr('opacity', 0.3);
      g.append('text')
        .attr('x', -4).attr('y', yScale(c))
        .attr('text-anchor', 'end').attr('dominant-baseline', 'middle')
        .attr('fill', CLASS_COLORS[c]).attr('font-size', '9px')
        .text(CLASS_NAMES[c].slice(0, 3));
    });

    // Area fill
    const area = d3.area()
      .x((d, i) => xScale(i))
      .y0(h)
      .y1(d => yScale(d.predicted_class ?? 1))
      .curve(d3.curveStepAfter);
    g.append('path')
      .datum(trajectory)
      .attr('d', area)
      .attr('fill', '#58a6ff').attr('opacity', 0.08);

    // Line
    const line = d3.line()
      .x((d, i) => xScale(i))
      .y(d => yScale(d.predicted_class ?? 1))
      .curve(d3.curveStepAfter);
    g.append('path')
      .datum(trajectory)
      .attr('d', line)
      .attr('fill', 'none')
      .attr('stroke', '#58a6ff')
      .attr('stroke-width', 2);

    // Confidence dots
    trajectory.forEach((d, i) => {
      g.append('circle')
        .attr('cx', xScale(i)).attr('cy', yScale(d.predicted_class ?? 1))
        .attr('r', 3)
        .attr('fill', CLASS_COLORS[d.predicted_class ?? 1])
        .attr('opacity', d.confidence ?? 0.5);
    });

    // X axis (time)
    const xAxis = d3.axisBottom(xScale).ticks(5).tickFormat(i => `${-Math.round((trajectory.length - 1 - i) * 0.5)}s`);
    g.append('g').attr('transform', `translate(0,${h})`).call(xAxis)
      .selectAll('text, line, path').attr('stroke', '#30363d').attr('fill', '#8b949e');

  }, [trajectory]);

  return <svg ref={svgRef} style={{ display: 'block', width: '100%' }} />;
}
