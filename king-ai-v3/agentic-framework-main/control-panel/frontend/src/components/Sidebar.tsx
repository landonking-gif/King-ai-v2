import { useState } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import {
  LayoutDashboard,
  Workflow,
  CheckCircle,
  BarChart3,
  Settings,
  ChevronLeft,
  ChevronRight
} from 'lucide-react'

const Sidebar = () => {
  const [collapsed, setCollapsed] = useState(false)
  const navigate = useNavigate()
  const location = useLocation()

  const menuItems = [
    { icon: LayoutDashboard, label: 'Command Center', path: '/', active: location.pathname === '/' },
    { icon: Workflow, label: 'Workflow Studio', path: '/workflow-studio', active: location.pathname === '/workflow-studio' },
    { icon: CheckCircle, label: 'Approval Center', path: '/approvals', active: location.pathname === '/approvals' },
    { icon: BarChart3, label: 'Analytics', path: '/analytics', active: location.pathname === '/analytics' },
    { icon: Settings, label: 'Settings', path: '/settings', active: location.pathname === '/settings' }
  ]

  return (
    <aside className={`bg-white dark:bg-gray-800 shadow-sm border-r border-gray-200 dark:border-gray-700 transition-all duration-300 ${
      collapsed ? 'w-16' : 'w-64'
    }`}>
      <div className="flex items-center justify-between p-4">
        {!collapsed && (
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
            Navigation
          </h2>
        )}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
        >
          {collapsed ? <ChevronRight size={20} /> : <ChevronLeft size={20} />}
        </button>
      </div>
      <nav className="px-2">
        {menuItems.map((item, index) => (
          <button
            key={index}
            onClick={() => navigate(item.path)}
            className={`w-full flex items-center px-3 py-2 mb-1 rounded-lg transition-colors ${
              item.active
                ? 'bg-blue-50 text-blue-700 dark:bg-blue-900 dark:text-blue-300'
                : 'text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-700'
            }`}
          >
            <item.icon size={20} className="mr-3" />
            {!collapsed && <span>{item.label}</span>}
          </button>
        ))}
      </nav>
    </aside>
  )
}

export default Sidebar