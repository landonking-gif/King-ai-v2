# Implementation Plan Part 22: Dashboard React Components

| Field | Value |
|-------|-------|
| Module | Dashboard UI Components |
| Priority | High |
| Estimated Effort | 6-7 hours |
| Dependencies | Part 3 (API), Vite/React setup |

---

## 1. Scope

This module implements the React dashboard components:

- **Layout Components** - Sidebar, header, navigation
- **Business Cards** - Display business status and metrics
- **Charts & Visualizations** - Revenue, growth, performance charts
- **Data Tables** - Sortable, filterable data displays
- **Status Indicators** - Health scores, alerts, progress

---

## 2. Tasks

### Task 22.1: Layout Components

**File: `dashboard/src/components/Layout/Sidebar.jsx`**

```jsx
import { useState } from 'react';
import { NavLink } from 'react-router-dom';
import './Sidebar.css';

const menuItems = [
  { path: '/', icon: '📊', label: 'Dashboard' },
  { path: '/businesses', icon: '🏢', label: 'Businesses' },
  { path: '/portfolio', icon: '📁', label: 'Portfolio' },
  { path: '/playbooks', icon: '📋', label: 'Playbooks' },
  { path: '/analytics', icon: '📈', label: 'Analytics' },
  { path: '/approvals', icon: '✅', label: 'Approvals' },
  { path: '/settings', icon: '⚙️', label: 'Settings' },
];

export function Sidebar({ collapsed, onToggle }) {
  return (
    <aside className={`sidebar ${collapsed ? 'collapsed' : ''}`}>
      <div className="sidebar-header">
        <h1 className="logo">{collapsed ? 'K' : 'King AI'}</h1>
        <button className="toggle-btn" onClick={onToggle}>
          {collapsed ? '→' : '←'}
        </button>
      </div>
      
      <nav className="sidebar-nav">
        {menuItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) => 
              `nav-item ${isActive ? 'active' : ''}`
            }
          >
            <span className="nav-icon">{item.icon}</span>
            {!collapsed && <span className="nav-label">{item.label}</span>}
          </NavLink>
        ))}
      </nav>
      
      <div className="sidebar-footer">
        {!collapsed && <span className="version">v2.0.0</span>}
      </div>
    </aside>
  );
}
```

**File: `dashboard/src/components/Layout/Header.jsx`**

```jsx
import { useState } from 'react';
import './Header.css';

export function Header({ user, notifications = [] }) {
  const [showNotifications, setShowNotifications] = useState(false);
  const [showProfile, setShowProfile] = useState(false);
  
  const unreadCount = notifications.filter(n => !n.read).length;
  
  return (
    <header className="main-header">
      <div className="header-search">
        <input
          type="text"
          placeholder="Search businesses, playbooks..."
          className="search-input"
        />
      </div>
      
      <div className="header-actions">
        <div className="notification-wrapper">
          <button
            className="icon-btn"
            onClick={() => setShowNotifications(!showNotifications)}
          >
            🔔
            {unreadCount > 0 && (
              <span className="badge">{unreadCount}</span>
            )}
          </button>
          
          {showNotifications && (
            <div className="dropdown notifications-dropdown">
              <h4>Notifications</h4>
              {notifications.length === 0 ? (
                <p className="empty">No notifications</p>
              ) : (
                notifications.slice(0, 5).map((n, i) => (
                  <div key={i} className={`notification ${n.read ? '' : 'unread'}`}>
                    <span className="notif-icon">{n.icon || '📌'}</span>
                    <div className="notif-content">
                      <p>{n.message}</p>
                      <small>{n.time}</small>
                    </div>
                  </div>
                ))
              )}
            </div>
          )}
        </div>
        
        <div className="profile-wrapper">
          <button
            className="profile-btn"
            onClick={() => setShowProfile(!showProfile)}
          >
            <img
              src={user?.avatar || '/default-avatar.png'}
              alt="Profile"
              className="avatar"
            />
          </button>
          
          {showProfile && (
            <div className="dropdown profile-dropdown">
              <div className="profile-info">
                <strong>{user?.name || 'User'}</strong>
                <small>{user?.email}</small>
              </div>
              <hr />
              <a href="/settings">Settings</a>
              <a href="/logout">Logout</a>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
```

**File: `dashboard/src/components/Layout/MainLayout.jsx`**

```jsx
import { useState } from 'react';
import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import './MainLayout.css';

export function MainLayout({ user }) {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [notifications, setNotifications] = useState([]);
  
  return (
    <div className={`app-layout ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`}>
      <Sidebar
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
      />
      
      <div className="main-content">
        <Header user={user} notifications={notifications} />
        
        <main className="page-content">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
```

---

### Task 22.2: Business Components

**File: `dashboard/src/components/Business/BusinessCard.jsx`**

```jsx
import { HealthBadge } from '../Common/HealthBadge';
import { ProgressBar } from '../Common/ProgressBar';
import './BusinessCard.css';

export function BusinessCard({ business, onClick }) {
  const {
    id,
    name,
    type,
    stage,
    health_score,
    revenue,
    growth_rate,
    active_tasks,
  } = business;
  
  const stageColors = {
    ideation: '#9b59b6',
    validation: '#3498db',
    launch: '#e67e22',
    growth: '#27ae60',
    scaling: '#2ecc71',
    maturity: '#1abc9c',
  };
  
  return (
    <div className="business-card" onClick={() => onClick?.(id)}>
      <div className="card-header">
        <h3 className="business-name">{name}</h3>
        <span
          className="stage-badge"
          style={{ backgroundColor: stageColors[stage] || '#95a5a6' }}
        >
          {stage}
        </span>
      </div>
      
      <div className="card-body">
        <div className="metric-row">
          <span className="metric-label">Health</span>
          <HealthBadge score={health_score} />
        </div>
        
        <div className="metric-row">
          <span className="metric-label">Revenue</span>
          <span className="metric-value">${revenue?.toLocaleString()}</span>
        </div>
        
        <div className="metric-row">
          <span className="metric-label">Growth</span>
          <span className={`metric-value ${growth_rate >= 0 ? 'positive' : 'negative'}`}>
            {growth_rate >= 0 ? '+' : ''}{growth_rate}%
          </span>
        </div>
        
        {active_tasks > 0 && (
          <div className="active-tasks">
            <span>🔄 {active_tasks} active tasks</span>
          </div>
        )}
      </div>
      
      <div className="card-footer">
        <span className="business-type">{type}</span>
      </div>
    </div>
  );
}
```

**File: `dashboard/src/components/Business/BusinessList.jsx`**

```jsx
import { useState, useMemo } from 'react';
import { BusinessCard } from './BusinessCard';
import './BusinessList.css';

export function BusinessList({ businesses, onSelect }) {
  const [filter, setFilter] = useState('all');
  const [sortBy, setSortBy] = useState('name');
  const [searchTerm, setSearchTerm] = useState('');
  
  const filteredBusinesses = useMemo(() => {
    let result = [...businesses];
    
    // Filter by stage
    if (filter !== 'all') {
      result = result.filter(b => b.stage === filter);
    }
    
    // Filter by search
    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      result = result.filter(b =>
        b.name.toLowerCase().includes(term) ||
        b.type.toLowerCase().includes(term)
      );
    }
    
    // Sort
    result.sort((a, b) => {
      switch (sortBy) {
        case 'revenue':
          return (b.revenue || 0) - (a.revenue || 0);
        case 'health':
          return (b.health_score || 0) - (a.health_score || 0);
        case 'growth':
          return (b.growth_rate || 0) - (a.growth_rate || 0);
        default:
          return a.name.localeCompare(b.name);
      }
    });
    
    return result;
  }, [businesses, filter, sortBy, searchTerm]);
  
  const stages = ['all', 'ideation', 'validation', 'launch', 'growth', 'scaling'];
  
  return (
    <div className="business-list">
      <div className="list-controls">
        <input
          type="text"
          placeholder="Search businesses..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="search-input"
        />
        
        <div className="filter-group">
          {stages.map(stage => (
            <button
              key={stage}
              className={`filter-btn ${filter === stage ? 'active' : ''}`}
              onClick={() => setFilter(stage)}
            >
              {stage === 'all' ? 'All' : stage}
            </button>
          ))}
        </div>
        
        <select
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value)}
          className="sort-select"
        >
          <option value="name">Name</option>
          <option value="revenue">Revenue</option>
          <option value="health">Health</option>
          <option value="growth">Growth</option>
        </select>
      </div>
      
      <div className="cards-grid">
        {filteredBusinesses.map(business => (
          <BusinessCard
            key={business.id}
            business={business}
            onClick={onSelect}
          />
        ))}
        
        {filteredBusinesses.length === 0 && (
          <div className="empty-state">
            <p>No businesses found</p>
          </div>
        )}
      </div>
    </div>
  );
}
```

---

### Task 22.3: Chart Components

**File: `dashboard/src/components/Charts/RevenueChart.jsx`**

```jsx
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
```

**File: `dashboard/src/components/Charts/MetricCard.jsx`**

```jsx
import './Charts.css';

export function MetricCard({
  title,
  value,
  change,
  changeLabel,
  icon,
  trend,
}) {
  const isPositive = change >= 0;
  const trendIcon = trend === 'up' ? '↑' : trend === 'down' ? '↓' : '→';
  
  return (
    <div className="metric-card">
      <div className="metric-header">
        {icon && <span className="metric-icon">{icon}</span>}
        <span className="metric-title">{title}</span>
      </div>
      
      <div className="metric-value-large">{value}</div>
      
      {change !== undefined && (
        <div className={`metric-change ${isPositive ? 'positive' : 'negative'}`}>
          <span className="trend-icon">{trendIcon}</span>
          <span>{isPositive ? '+' : ''}{change}%</span>
          {changeLabel && <span className="change-label">{changeLabel}</span>}
        </div>
      )}
    </div>
  );
}
```

**File: `dashboard/src/components/Charts/DonutChart.jsx`**

```jsx
import { useMemo } from 'react';
import './Charts.css';

export function DonutChart({ data, size = 150, thickness = 20 }) {
  const total = useMemo(() => 
    data.reduce((sum, d) => sum + d.value, 0),
    [data]
  );
  
  const segments = useMemo(() => {
    let currentAngle = -90; // Start from top
    
    return data.map((d) => {
      const angle = (d.value / total) * 360;
      const startAngle = currentAngle;
      currentAngle += angle;
      
      return {
        ...d,
        startAngle,
        endAngle: currentAngle,
        percent: ((d.value / total) * 100).toFixed(1),
      };
    });
  }, [data, total]);
  
  const center = size / 2;
  const radius = (size - thickness) / 2;
  
  const getArcPath = (startAngle, endAngle) => {
    const startRad = (startAngle * Math.PI) / 180;
    const endRad = (endAngle * Math.PI) / 180;
    
    const x1 = center + radius * Math.cos(startRad);
    const y1 = center + radius * Math.sin(startRad);
    const x2 = center + radius * Math.cos(endRad);
    const y2 = center + radius * Math.sin(endRad);
    
    const largeArc = endAngle - startAngle > 180 ? 1 : 0;
    
    return `M ${x1} ${y1} A ${radius} ${radius} 0 ${largeArc} 1 ${x2} ${y2}`;
  };
  
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
```

---

### Task 22.4: Common Components

**File: `dashboard/src/components/Common/HealthBadge.jsx`**

```jsx
import './Common.css';

export function HealthBadge({ score, showLabel = true }) {
  const getColor = (score) => {
    if (score >= 80) return '#27ae60';
    if (score >= 60) return '#f39c12';
    if (score >= 40) return '#e67e22';
    return '#e74c3c';
  };
  
  const getLabel = (score) => {
    if (score >= 80) return 'Excellent';
    if (score >= 60) return 'Good';
    if (score >= 40) return 'Fair';
    return 'Poor';
  };
  
  return (
    <div className="health-badge" style={{ '--health-color': getColor(score) }}>
      <span className="health-score">{score}</span>
      {showLabel && <span className="health-label">{getLabel(score)}</span>}
    </div>
  );
}
```

**File: `dashboard/src/components/Common/ProgressBar.jsx`**

```jsx
import './Common.css';

export function ProgressBar({ value, max = 100, label, showPercent = true, color }) {
  const percent = Math.min(100, (value / max) * 100);
  
  const getAutoColor = (pct) => {
    if (pct >= 75) return '#27ae60';
    if (pct >= 50) return '#3498db';
    if (pct >= 25) return '#f39c12';
    return '#e74c3c';
  };
  
  const barColor = color || getAutoColor(percent);
  
  return (
    <div className="progress-container">
      {label && <span className="progress-label">{label}</span>}
      
      <div className="progress-bar">
        <div
          className="progress-fill"
          style={{
            width: `${percent}%`,
            backgroundColor: barColor,
          }}
        />
      </div>
      
      {showPercent && (
        <span className="progress-percent">{percent.toFixed(0)}%</span>
      )}
    </div>
  );
}
```

**File: `dashboard/src/components/Common/DataTable.jsx`**

```jsx
import { useState, useMemo } from 'react';
import './Common.css';

export function DataTable({ columns, data, onRowClick, pageSize = 10 }) {
  const [sortColumn, setSortColumn] = useState(null);
  const [sortDir, setSortDir] = useState('asc');
  const [page, setPage] = useState(0);
  
  const sortedData = useMemo(() => {
    if (!sortColumn) return data;
    
    return [...data].sort((a, b) => {
      const aVal = a[sortColumn];
      const bVal = b[sortColumn];
      
      if (typeof aVal === 'number') {
        return sortDir === 'asc' ? aVal - bVal : bVal - aVal;
      }
      
      const cmp = String(aVal).localeCompare(String(bVal));
      return sortDir === 'asc' ? cmp : -cmp;
    });
  }, [data, sortColumn, sortDir]);
  
  const pagedData = useMemo(() => {
    const start = page * pageSize;
    return sortedData.slice(start, start + pageSize);
  }, [sortedData, page, pageSize]);
  
  const totalPages = Math.ceil(data.length / pageSize);
  
  const handleSort = (column) => {
    if (sortColumn === column) {
      setSortDir(sortDir === 'asc' ? 'desc' : 'asc');
    } else {
      setSortColumn(column);
      setSortDir('asc');
    }
  };
  
  return (
    <div className="data-table-container">
      <table className="data-table">
        <thead>
          <tr>
            {columns.map((col) => (
              <th
                key={col.key}
                onClick={() => col.sortable !== false && handleSort(col.key)}
                className={col.sortable !== false ? 'sortable' : ''}
              >
                {col.label}
                {sortColumn === col.key && (
                  <span className="sort-icon">
                    {sortDir === 'asc' ? '↑' : '↓'}
                  </span>
                )}
              </th>
            ))}
          </tr>
        </thead>
        
        <tbody>
          {pagedData.map((row, i) => (
            <tr
              key={row.id || i}
              onClick={() => onRowClick?.(row)}
              className={onRowClick ? 'clickable' : ''}
            >
              {columns.map((col) => (
                <td key={col.key}>
                  {col.render ? col.render(row[col.key], row) : row[col.key]}
                </td>
              ))}
            </tr>
          ))}
          
          {pagedData.length === 0 && (
            <tr>
              <td colSpan={columns.length} className="empty-row">
                No data available
              </td>
            </tr>
          )}
        </tbody>
      </table>
      
      {totalPages > 1 && (
        <div className="table-pagination">
          <button
            disabled={page === 0}
            onClick={() => setPage(page - 1)}
          >
            Previous
          </button>
          
          <span className="page-info">
            Page {page + 1} of {totalPages}
          </span>
          
          <button
            disabled={page >= totalPages - 1}
            onClick={() => setPage(page + 1)}
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}
```

**File: `dashboard/src/components/Common/StatusIndicator.jsx`**

```jsx
import './Common.css';

const statusConfig = {
  running: { color: '#3498db', icon: '🔄', label: 'Running' },
  completed: { color: '#27ae60', icon: '✓', label: 'Completed' },
  failed: { color: '#e74c3c', icon: '✗', label: 'Failed' },
  pending: { color: '#f39c12', icon: '⏳', label: 'Pending' },
  paused: { color: '#95a5a6', icon: '⏸', label: 'Paused' },
  active: { color: '#27ae60', icon: '●', label: 'Active' },
  inactive: { color: '#95a5a6', icon: '○', label: 'Inactive' },
};

export function StatusIndicator({ status, showLabel = true, size = 'medium' }) {
  const config = statusConfig[status] || statusConfig.pending;
  
  return (
    <span className={`status-indicator ${size}`} style={{ '--status-color': config.color }}>
      <span className="status-icon">{config.icon}</span>
      {showLabel && <span className="status-label">{config.label}</span>}
    </span>
  );
}
```

---

### Task 22.5: Styles

**File: `dashboard/src/components/Common/Common.css`**

```css
/* Health Badge */
.health-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 8px;
  border-radius: 4px;
  background: color-mix(in srgb, var(--health-color) 15%, transparent);
  border: 1px solid var(--health-color);
}

.health-score {
  font-weight: 600;
  color: var(--health-color);
}

.health-label {
  font-size: 0.85em;
  color: var(--health-color);
}

/* Progress Bar */
.progress-container {
  display: flex;
  align-items: center;
  gap: 8px;
}

.progress-bar {
  flex: 1;
  height: 8px;
  background: #e0e0e0;
  border-radius: 4px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  transition: width 0.3s ease;
}

.progress-percent {
  font-size: 0.85em;
  min-width: 40px;
  text-align: right;
}

/* Data Table */
.data-table-container {
  overflow-x: auto;
}

.data-table {
  width: 100%;
  border-collapse: collapse;
}

.data-table th,
.data-table td {
  padding: 12px;
  text-align: left;
  border-bottom: 1px solid #e0e0e0;
}

.data-table th {
  background: #f5f5f5;
  font-weight: 600;
}

.data-table th.sortable {
  cursor: pointer;
}

.data-table th.sortable:hover {
  background: #eee;
}

.data-table tr.clickable {
  cursor: pointer;
}

.data-table tr.clickable:hover {
  background: #f9f9f9;
}

.table-pagination {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 16px;
  padding: 16px;
}

/* Status Indicator */
.status-indicator {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  color: var(--status-color);
}

.status-indicator.small { font-size: 0.8em; }
.status-indicator.large { font-size: 1.2em; }

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.status-indicator .status-icon {
  animation: pulse 2s infinite;
}
```

---

### Task 22.6: Component Index

**File: `dashboard/src/components/index.js`**

```javascript
// Layout
export { Sidebar } from './Layout/Sidebar';
export { Header } from './Layout/Header';
export { MainLayout } from './Layout/MainLayout';

// Business
export { BusinessCard } from './Business/BusinessCard';
export { BusinessList } from './Business/BusinessList';

// Charts
export { RevenueChart } from './Charts/RevenueChart';
export { MetricCard } from './Charts/MetricCard';
export { DonutChart } from './Charts/DonutChart';

// Common
export { HealthBadge } from './Common/HealthBadge';
export { ProgressBar } from './Common/ProgressBar';
export { DataTable } from './Common/DataTable';
export { StatusIndicator } from './Common/StatusIndicator';
```

---

## 3. Acceptance Criteria

| Criteria | Validation |
|----------|------------|
| Layout renders | Sidebar, header, main content |
| Business cards | Display metrics and status |
| Charts work | Revenue, donut charts render |
| Tables sortable | Sort and paginate data |
| Status indicators | Show correct colors/icons |
| Responsive | Works on different screen sizes |

---

## 4. File Summary

| File | Purpose |
|------|---------|
| `components/Layout/Sidebar.jsx` | Navigation sidebar |
| `components/Layout/Header.jsx` | Top header with notifications |
| `components/Layout/MainLayout.jsx` | Main app layout wrapper |
| `components/Business/BusinessCard.jsx` | Business summary card |
| `components/Business/BusinessList.jsx` | Filterable business list |
| `components/Charts/RevenueChart.jsx` | Line chart for revenue |
| `components/Charts/MetricCard.jsx` | KPI metric display |
| `components/Charts/DonutChart.jsx` | Donut/pie chart |
| `components/Common/HealthBadge.jsx` | Health score badge |
| `components/Common/ProgressBar.jsx` | Progress indicator |
| `components/Common/DataTable.jsx` | Sortable data table |
| `components/Common/StatusIndicator.jsx` | Status display |
| `components/Common/Common.css` | Shared styles |
| `components/index.js` | Component exports |
