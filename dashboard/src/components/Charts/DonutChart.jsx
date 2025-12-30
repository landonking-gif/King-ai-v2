import { useMemo, useCallback } from 'react';
import './Charts.css';

export function DonutChart({ data, size = 150, thickness = 20 }) {
  const total = useMemo(() => 
    data.reduce((sum, d) => sum + d.value, 0),
    [data]
  );
  
  const segments = useMemo(() => {
    const result = [];
    let currentAngle = -90; // Start from top
    
    for (const d of data) {
      const angle = (d.value / total) * 360;
      const startAngle = currentAngle;
      const endAngle = currentAngle + angle;
      
      result.push({
        ...d,
        startAngle,
        endAngle,
        percent: ((d.value / total) * 100).toFixed(1),
      });
      
      currentAngle = endAngle;
    }
    
    return result;
  }, [data, total]);
  
  const center = size / 2;
  const radius = (size - thickness) / 2;
  
  const getArcPath = useCallback((startAngle, endAngle) => {
    const startRad = (startAngle * Math.PI) / 180;
    const endRad = (endAngle * Math.PI) / 180;
    
    const x1 = center + radius * Math.cos(startRad);
    const y1 = center + radius * Math.sin(startRad);
    const x2 = center + radius * Math.cos(endRad);
    const y2 = center + radius * Math.sin(endRad);
    
    const largeArc = endAngle - startAngle > 180 ? 1 : 0;
    
    return `M ${x1} ${y1} A ${radius} ${radius} 0 ${largeArc} 1 ${x2} ${y2}`;
  }, [center, radius]);
  
  return (
    <div className="donut-chart-container">
      <svg width={size} height={size} className="donut-chart">
        {segments.map((seg, i) => (
          <path
            key={i}
            d={getArcPath(seg.startAngle, seg.endAngle - 0.5)}
            fill="none"
            stroke={seg.color}
            strokeWidth={thickness}
            className="donut-segment"
          >
            <title>{seg.label}: {seg.percent}%</title>
          </path>
        ))}
        
        <text
          x={center}
          y={center}
          textAnchor="middle"
          dominantBaseline="middle"
          className="donut-center-text"
        >
          {total}
        </text>
      </svg>
      
      <div className="donut-legend">
        {segments.map((seg, i) => (
          <div key={i} className="legend-item">
            <span
              className="legend-color"
              style={{ backgroundColor: seg.color }}
            />
            <span className="legend-label">{seg.label}</span>
            <span className="legend-value">{seg.percent}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}
