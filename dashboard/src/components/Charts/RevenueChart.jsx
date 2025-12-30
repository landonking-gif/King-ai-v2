import { useMemo } from 'react';
import './Charts.css';

export function RevenueChart({ data, height = 200 }) {
  const chartData = useMemo(() => {
    if (!data || data.length === 0) return { points: [], max: 0 };
    
    const max = Math.max(...data.map(d => d.value));
    const points = data.map((d, i) => ({
      x: (i / (data.length - 1)) * 100,
      y: 100 - (d.value / max) * 100,
      value: d.value,
      label: d.label,
    }));
    
    return { points, max };
  }, [data]);
  
  const pathD = chartData.points.length > 1
    ? `M ${chartData.points.map(p => `${p.x},${p.y}`).join(' L ')}`
    : '';
  
  const areaD = pathD
    ? `${pathD} L 100,100 L 0,100 Z`
    : '';
  
  return (
    <div className="chart-container" style={{ height }}>
      <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="line-chart">
        {/* Grid lines */}
        {[0, 25, 50, 75, 100].map(y => (
          <line
            key={y}
            x1="0" y1={y} x2="100" y2={y}
            className="grid-line"
          />
        ))}
        
        {/* Area fill */}
        <path d={areaD} className="chart-area" />
        
        {/* Line */}
        <path d={pathD} className="chart-line" />
        
        {/* Data points */}
        {chartData.points.map((p, i) => (
          <circle
            key={i}
            cx={p.x}
            cy={p.y}
            r="1.5"
            className="chart-point"
          >
            <title>{p.label}: ${p.value.toLocaleString()}</title>
          </circle>
        ))}
      </svg>
      
      <div className="chart-labels">
        {data?.slice(0, 6).map((d, i) => (
          <span key={i} className="chart-label">{d.label}</span>
        ))}
      </div>
    </div>
  );
}
