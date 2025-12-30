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
