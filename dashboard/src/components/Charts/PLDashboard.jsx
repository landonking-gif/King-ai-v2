import { useState, useMemo, useCallback } from 'react';
import { MetricCard } from './MetricCard';
import { RevenueChart } from './RevenueChart';
import { DonutChart } from './DonutChart';
import './Charts.css';
import './PLDashboard.css';

/**
 * P&L Time-Series Dashboard Component
 * 
 * Displays comprehensive profit & loss data with:
 * - Revenue, expenses, and profit trends
 * - Margin analysis
 * - Cost breakdown
 * - Period comparison
 */
export function PLDashboard({ 
  data, 
  period = '30d',
  onPeriodChange,
  businessId = null,
  showComparison = true 
}) {
  const [selectedMetric, setSelectedMetric] = useState('profit');
  const [chartView, setChartView] = useState('line'); // line, bar, area
  
  // Calculate aggregate metrics
  const metrics = useMemo(() => {
    if (!data || !data.timeSeries || data.timeSeries.length === 0) {
      return {
        totalRevenue: 0,
        totalExpenses: 0,
        netProfit: 0,
        profitMargin: 0,
        avgDailyRevenue: 0,
        revenueGrowth: 0,
        expenseRatio: 0,
      };
    }
    
    const series = data.timeSeries;
    const totalRevenue = series.reduce((sum, d) => sum + (d.revenue || 0), 0);
    const totalExpenses = series.reduce((sum, d) => sum + (d.expenses || 0), 0);
    const netProfit = totalRevenue - totalExpenses;
    const profitMargin = totalRevenue > 0 ? (netProfit / totalRevenue) * 100 : 0;
    const avgDailyRevenue = totalRevenue / series.length;
    
    // Calculate growth (compare first half to second half)
    const midpoint = Math.floor(series.length / 2);
    const firstHalfRevenue = series.slice(0, midpoint).reduce((sum, d) => sum + (d.revenue || 0), 0);
    const secondHalfRevenue = series.slice(midpoint).reduce((sum, d) => sum + (d.revenue || 0), 0);
    const revenueGrowth = firstHalfRevenue > 0 
      ? ((secondHalfRevenue - firstHalfRevenue) / firstHalfRevenue) * 100 
      : 0;
    
    const expenseRatio = totalRevenue > 0 ? (totalExpenses / totalRevenue) * 100 : 0;
    
    return {
      totalRevenue,
      totalExpenses,
      netProfit,
      profitMargin,
      avgDailyRevenue,
      revenueGrowth,
      expenseRatio,
    };
  }, [data]);
  
  // Prepare chart data based on selected metric
  const chartData = useMemo(() => {
    if (!data || !data.timeSeries) return [];
    
    return data.timeSeries.map(d => {
      let value;
      switch (selectedMetric) {
        case 'revenue':
          value = d.revenue || 0;
          break;
        case 'expenses':
          value = d.expenses || 0;
          break;
        case 'profit':
          value = (d.revenue || 0) - (d.expenses || 0);
          break;
        case 'margin':
          value = d.revenue > 0 ? ((d.revenue - d.expenses) / d.revenue) * 100 : 0;
          break;
        default:
          value = d.revenue || 0;
      }
      
      return {
        label: formatDateLabel(d.date, period),
        value,
        date: d.date,
        revenue: d.revenue,
        expenses: d.expenses,
      };
    });
  }, [data, selectedMetric, period]);
  
  // Expense breakdown for donut chart
  const expenseBreakdown = useMemo(() => {
    if (!data || !data.expenseCategories) return [];
    
    return data.expenseCategories.map(cat => ({
      label: cat.name,
      value: cat.amount,
      color: getCategoryColor(cat.name),
    }));
  }, [data]);
  
  // Period options
  const periodOptions = [
    { value: '7d', label: '7 Days' },
    { value: '30d', label: '30 Days' },
    { value: '90d', label: '90 Days' },
    { value: 'ytd', label: 'Year to Date' },
    { value: '1y', label: '1 Year' },
  ];
  
  const handlePeriodChange = useCallback((e) => {
    if (onPeriodChange) {
      onPeriodChange(e.target.value);
    }
  }, [onPeriodChange]);
  
  return (
    <div className="pl-dashboard">
      {/* Header with period selector */}
      <div className="pl-dashboard-header">
        <h2>Profit & Loss Analysis</h2>
        <div className="pl-controls">
          <select 
            value={period} 
            onChange={handlePeriodChange}
            className="period-select"
          >
            {periodOptions.map(opt => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
        </div>
      </div>
      
      {/* Key Metrics */}
      <div className="pl-metrics-grid">
        <MetricCard
          title="Total Revenue"
          value={`$${formatCurrency(metrics.totalRevenue)}`}
          trend={metrics.revenueGrowth}
          trendLabel="vs prev period"
          icon="💰"
          onClick={() => setSelectedMetric('revenue')}
          active={selectedMetric === 'revenue'}
        />
        <MetricCard
          title="Total Expenses"
          value={`$${formatCurrency(metrics.totalExpenses)}`}
          trend={-metrics.expenseRatio}
          trendLabel="of revenue"
          icon="📉"
          onClick={() => setSelectedMetric('expenses')}
          active={selectedMetric === 'expenses'}
        />
        <MetricCard
          title="Net Profit"
          value={`$${formatCurrency(metrics.netProfit)}`}
          trend={metrics.profitMargin}
          trendLabel="margin"
          icon={metrics.netProfit >= 0 ? "📈" : "📉"}
          positive={metrics.netProfit >= 0}
          onClick={() => setSelectedMetric('profit')}
          active={selectedMetric === 'profit'}
        />
        <MetricCard
          title="Profit Margin"
          value={`${metrics.profitMargin.toFixed(1)}%`}
          trend={0}
          icon="🎯"
          onClick={() => setSelectedMetric('margin')}
          active={selectedMetric === 'margin'}
        />
      </div>
      
      {/* Main Chart Section */}
      <div className="pl-chart-section">
        <div className="chart-header">
          <h3>{getMetricLabel(selectedMetric)} Over Time</h3>
          <div className="chart-view-toggle">
            <button 
              className={chartView === 'line' ? 'active' : ''} 
              onClick={() => setChartView('line')}
            >
              Line
            </button>
            <button 
              className={chartView === 'bar' ? 'active' : ''} 
              onClick={() => setChartView('bar')}
            >
              Bar
            </button>
            <button 
              className={chartView === 'area' ? 'active' : ''} 
              onClick={() => setChartView('area')}
            >
              Area
            </button>
          </div>
        </div>
        
        <div className="pl-main-chart">
          {chartView === 'line' && (
            <RevenueChart data={chartData} height={300} />
          )}
          {chartView === 'bar' && (
            <BarChart data={chartData} height={300} />
          )}
          {chartView === 'area' && (
            <AreaChart data={chartData} height={300} />
          )}
        </div>
      </div>
      
      {/* Secondary Charts */}
      <div className="pl-secondary-charts">
        {/* Expense Breakdown */}
        <div className="chart-card">
          <h3>Expense Breakdown</h3>
          <DonutChart 
            data={expenseBreakdown} 
            size={200}
            showLegend={true}
          />
        </div>
        
        {/* Revenue vs Expenses Comparison */}
        <div className="chart-card">
          <h3>Revenue vs Expenses</h3>
          <ComparisonChart 
            revenue={data?.timeSeries?.map(d => ({ 
              label: formatDateLabel(d.date, period), 
              value: d.revenue 
            })) || []}
            expenses={data?.timeSeries?.map(d => ({ 
              label: formatDateLabel(d.date, period), 
              value: d.expenses 
            })) || []}
            height={200}
          />
        </div>
        
        {/* Profit Trend with Target */}
        <div className="chart-card">
          <h3>Profit vs Target</h3>
          <ProfitTargetChart 
            actual={metrics.netProfit}
            target={data?.profitTarget || 0}
            previousPeriod={data?.previousPeriodProfit || 0}
          />
        </div>
      </div>
      
      {/* Detailed P&L Table */}
      {showComparison && (
        <div className="pl-table-section">
          <h3>Detailed P&L Statement</h3>
          <PLTable 
            data={data}
            metrics={metrics}
          />
        </div>
      )}
    </div>
  );
}

// Helper Components

function BarChart({ data, height = 200 }) {
  if (!data || data.length === 0) return <div className="no-data">No data</div>;
  
  const max = Math.max(...data.map(d => Math.abs(d.value)));
  
  return (
    <div className="bar-chart" style={{ height }}>
      <div className="bar-chart-container">
        {data.map((d, i) => (
          <div key={i} className="bar-wrapper">
            <div 
              className={`bar ${d.value >= 0 ? 'positive' : 'negative'}`}
              style={{ 
                height: `${(Math.abs(d.value) / max) * 100}%`,
              }}
              title={`${d.label}: $${d.value.toLocaleString()}`}
            />
            <span className="bar-label">{d.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function AreaChart({ data, height = 200 }) {
  if (!data || data.length === 0) return <div className="no-data">No data</div>;
  
  const max = Math.max(...data.map(d => d.value));
  const points = data.map((d, i) => ({
    x: (i / (data.length - 1)) * 100,
    y: 100 - (d.value / max) * 100,
  }));
  
  const pathD = `M ${points.map(p => `${p.x},${p.y}`).join(' L ')}`;
  const areaD = `${pathD} L 100,100 L 0,100 Z`;
  
  return (
    <div className="area-chart" style={{ height }}>
      <svg viewBox="0 0 100 100" preserveAspectRatio="none">
        <defs>
          <linearGradient id="areaGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--accent-primary)" stopOpacity="0.4" />
            <stop offset="100%" stopColor="var(--accent-primary)" stopOpacity="0.05" />
          </linearGradient>
        </defs>
        <path d={areaD} fill="url(#areaGradient)" />
        <path d={pathD} fill="none" stroke="var(--accent-primary)" strokeWidth="2" />
      </svg>
    </div>
  );
}

function ComparisonChart({ revenue, expenses, height = 200 }) {
  if (!revenue || revenue.length === 0) return <div className="no-data">No data</div>;
  
  const maxValue = Math.max(
    ...revenue.map(d => d.value),
    ...expenses.map(d => d.value)
  );
  
  return (
    <div className="comparison-chart" style={{ height }}>
      <svg viewBox="0 0 100 100" preserveAspectRatio="none">
        {/* Revenue line */}
        <path 
          d={`M ${revenue.map((d, i) => 
            `${(i / (revenue.length - 1)) * 100},${100 - (d.value / maxValue) * 100}`
          ).join(' L ')}`}
          fill="none"
          stroke="var(--success)"
          strokeWidth="2"
        />
        {/* Expenses line */}
        <path 
          d={`M ${expenses.map((d, i) => 
            `${(i / (expenses.length - 1)) * 100},${100 - (d.value / maxValue) * 100}`
          ).join(' L ')}`}
          fill="none"
          stroke="var(--danger)"
          strokeWidth="2"
        />
      </svg>
      <div className="chart-legend">
        <span className="legend-item revenue">● Revenue</span>
        <span className="legend-item expenses">● Expenses</span>
      </div>
    </div>
  );
}

function ProfitTargetChart({ actual, target, previousPeriod }) {
  const maxValue = Math.max(actual, target, previousPeriod, 1);
  
  return (
    <div className="profit-target-chart">
      <div className="progress-bars">
        <div className="progress-bar-row">
          <span className="label">Actual</span>
          <div className="progress-track">
            <div 
              className={`progress-fill ${actual >= 0 ? 'positive' : 'negative'}`}
              style={{ width: `${(Math.abs(actual) / maxValue) * 100}%` }}
            />
          </div>
          <span className="value">${formatCurrency(actual)}</span>
        </div>
        <div className="progress-bar-row">
          <span className="label">Target</span>
          <div className="progress-track">
            <div 
              className="progress-fill target"
              style={{ width: `${(target / maxValue) * 100}%` }}
            />
          </div>
          <span className="value">${formatCurrency(target)}</span>
        </div>
        <div className="progress-bar-row">
          <span className="label">Previous</span>
          <div className="progress-track">
            <div 
              className="progress-fill previous"
              style={{ width: `${(Math.abs(previousPeriod) / maxValue) * 100}%` }}
            />
          </div>
          <span className="value">${formatCurrency(previousPeriod)}</span>
        </div>
      </div>
      <div className="target-status">
        {actual >= target ? (
          <span className="status-badge success">✓ Target Met</span>
        ) : (
          <span className="status-badge warning">
            ${formatCurrency(target - actual)} below target
          </span>
        )}
      </div>
    </div>
  );
}

function PLTable({ data, metrics }) {
  const rows = [
    { label: 'Gross Revenue', value: metrics.totalRevenue, type: 'revenue' },
    { label: 'Returns & Refunds', value: data?.refunds || 0, type: 'deduction' },
    { label: 'Net Revenue', value: metrics.totalRevenue - (data?.refunds || 0), type: 'subtotal' },
    { label: 'Cost of Goods Sold', value: data?.cogs || 0, type: 'expense' },
    { label: 'Gross Profit', value: (metrics.totalRevenue - (data?.refunds || 0)) - (data?.cogs || 0), type: 'subtotal' },
    { label: 'Marketing & Ads', value: data?.marketingExpenses || 0, type: 'expense' },
    { label: 'Platform Fees', value: data?.platformFees || 0, type: 'expense' },
    { label: 'Shipping Costs', value: data?.shippingCosts || 0, type: 'expense' },
    { label: 'Other Expenses', value: data?.otherExpenses || 0, type: 'expense' },
    { label: 'Total Operating Expenses', value: metrics.totalExpenses, type: 'subtotal' },
    { label: 'Net Profit', value: metrics.netProfit, type: 'total' },
  ];
  
  return (
    <table className="pl-table">
      <thead>
        <tr>
          <th>Item</th>
          <th>Amount</th>
          <th>% of Revenue</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row, i) => (
          <tr key={i} className={`row-${row.type}`}>
            <td>{row.label}</td>
            <td className={row.value < 0 ? 'negative' : ''}>
              ${formatCurrency(Math.abs(row.value))}
              {row.type === 'deduction' || row.type === 'expense' ? ' ' : ''}
            </td>
            <td>
              {metrics.totalRevenue > 0 
                ? `${((row.value / metrics.totalRevenue) * 100).toFixed(1)}%`
                : '-'
              }
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// Utility Functions

function formatCurrency(value) {
  if (value >= 1000000) {
    return `${(value / 1000000).toFixed(2)}M`;
  } else if (value >= 1000) {
    return `${(value / 1000).toFixed(1)}K`;
  }
  return value.toFixed(2);
}

function formatDateLabel(dateStr, period) {
  const date = new Date(dateStr);
  
  switch (period) {
    case '7d':
      return date.toLocaleDateString('en-US', { weekday: 'short' });
    case '30d':
      return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    case '90d':
    case 'ytd':
    case '1y':
      return date.toLocaleDateString('en-US', { month: 'short' });
    default:
      return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  }
}

function getMetricLabel(metric) {
  const labels = {
    revenue: 'Revenue',
    expenses: 'Expenses',
    profit: 'Net Profit',
    margin: 'Profit Margin',
  };
  return labels[metric] || metric;
}

function getCategoryColor(category) {
  const colors = {
    'Marketing': '#4f46e5',
    'Shipping': '#0891b2',
    'Platform Fees': '#7c3aed',
    'COGS': '#dc2626',
    'Advertising': '#ea580c',
    'Software': '#16a34a',
    'Other': '#64748b',
  };
  return colors[category] || '#64748b';
}

export default PLDashboard;
