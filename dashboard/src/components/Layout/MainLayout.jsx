import { useState } from 'react';
import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import './MainLayout.css';

export function MainLayout({ user }) {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [notifications] = useState([]);
  
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
