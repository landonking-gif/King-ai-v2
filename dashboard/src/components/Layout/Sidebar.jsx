import { NavLink } from 'react-router-dom';
import './Sidebar.css';

const menuItems = [
  { path: '/', icon: '📊', label: 'Dashboard', ariaLabel: 'Dashboard' },
  { path: '/businesses', icon: '🏢', label: 'Businesses', ariaLabel: 'Businesses' },
  { path: '/portfolio', icon: '📁', label: 'Portfolio', ariaLabel: 'Portfolio' },
  { path: '/playbooks', icon: '📋', label: 'Playbooks', ariaLabel: 'Playbooks' },
  { path: '/analytics', icon: '📈', label: 'Analytics', ariaLabel: 'Analytics' },
  { path: '/approvals', icon: '✅', label: 'Approvals', ariaLabel: 'Approvals' },
  { path: '/settings', icon: '⚙️', label: 'Settings', ariaLabel: 'Settings' },
];

export function Sidebar({ collapsed, onToggle }) {
  return (
    <aside className={`sidebar ${collapsed ? 'collapsed' : ''}`}>
      <div className="sidebar-header">
        <h1 className="logo">{collapsed ? 'K' : 'King AI'}</h1>
        <button className="toggle-btn" onClick={onToggle} aria-label="Toggle sidebar">
          {collapsed ? '→' : '←'}
        </button>
      </div>
      
      <nav className="sidebar-nav" aria-label="Main navigation">
        {menuItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) => 
              `nav-item ${isActive ? 'active' : ''}`
            }
            aria-label={item.ariaLabel}
          >
            <span className="nav-icon" aria-hidden="true">{item.icon}</span>
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
