import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Bell, User, LogOut } from 'lucide-react'

const Header = () => {
  const [user, setUser] = useState<any>(null)
  const navigate = useNavigate()

  useEffect(() => {
    fetchUser()
  }, [])

  const fetchUser = async () => {
    const token = localStorage.getItem('token')
    if (token) {
      try {
        const response = await fetch('/api/auth/me', {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        })
        if (response.ok) {
          const userData = await response.json()
          setUser(userData)
        }
      } catch (error) {
        console.error('Failed to fetch user:', error)
      }
    }
  }

  const handleLogout = async () => {
    try {
      const token = localStorage.getItem('token')
      await fetch('/api/auth/logout', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })
    } catch (error) {
      console.error('Logout error:', error)
    } finally {
      localStorage.removeItem('token')
      navigate('/login')
    }
  }

  return (
    <header className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
      <div className="flex items-center justify-between px-6 py-4">
        <div className="flex items-center">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            King AI v3 Control Panel
          </h1>
        </div>
        <div className="flex items-center space-x-4">
          <button className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300">
            <Bell size={20} />
          </button>
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2">
              <User size={20} className="text-gray-400" />
              <div className="text-right">
                <div className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  {user?.full_name || 'User'}
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400 capitalize">
                  {user?.role || 'Role'}
                </div>
              </div>
            </div>
            <button
              onClick={handleLogout}
              className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
              title="Logout"
            >
              <LogOut size={16} />
            </button>
          </div>
        </div>
      </div>
    </header>
  )
}

export default Header