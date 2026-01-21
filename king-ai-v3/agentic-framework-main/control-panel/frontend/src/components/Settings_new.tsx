import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Badge } from '../components/ui/badge'
import {
  Settings,
  User,
  Shield,
  Database,
  Key,
  Bell,
  Palette,
  Globe,
  Save,
  RefreshCw
} from 'lucide-react'

interface Setting {
  id: string
  category: string
  name: string
  description: string
  type: 'text' | 'number' | 'boolean' | 'select'
  value: any
  options?: string[]
  requiresRestart?: boolean
}

const SettingsPanel = () => {
  const [settings, setSettings] = useState<Setting[]>([])
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)

  // Static settings configuration
  const defaultSettings: Setting[] = [
    // System Settings
    {
      id: 'system-name',
      category: 'System',
      name: 'System Name',
      description: 'Display name for this King AI instance',
      type: 'text',
      value: 'King AI v3'
    },
    {
      id: 'system-version',
      category: 'System',
      name: 'Version',
      description: 'Current system version',
      type: 'text',
      value: '3.0.0'
    },
    {
      id: 'max-concurrent-jobs',
      category: 'System',
      name: 'Max Concurrent Jobs',
      description: 'Maximum number of jobs that can run simultaneously',
      type: 'number',
      value: 10
    },

    // API Settings
    {
      id: 'api-port',
      category: 'API',
      name: 'API Port',
      description: 'Port for the REST API server',
      type: 'number',
      value: 8000
    },
    {
      id: 'api-host',
      category: 'API',
      name: 'API Host',
      description: 'Host address for the API server',
      type: 'text',
      value: '0.0.0.0'
    },
    {
      id: 'cors-origins',
      category: 'API',
      name: 'CORS Origins',
      description: 'Allowed origins for CORS (comma-separated)',
      type: 'text',
      value: 'http://localhost:3000,http://localhost:5173'
    },

    // AI Settings
    {
      id: 'ollama-url',
      category: 'AI',
      name: 'Ollama Base URL',
      description: 'URL for the Ollama API server',
      type: 'text',
      value: 'http://localhost:11434'
    },
    {
      id: 'ollama-model',
      category: 'AI',
      name: 'Ollama Model',
      description: 'Default model to use for AI operations',
      type: 'text',
      value: 'llama2'
    },
    {
      id: 'temperature',
      category: 'AI',
      name: 'Temperature',
      description: 'Creativity level for AI responses (0.0-1.0)',
      type: 'number',
      value: 0.7
    },

    // Security Settings
    {
      id: 'jwt-secret',
      category: 'Security',
      name: 'JWT Secret Key',
      description: 'Secret key for JWT token generation',
      type: 'text',
      value: '••••••••••••••••',
      requiresRestart: true
    },
    {
      id: 'session-timeout',
      category: 'Security',
      name: 'Session Timeout (minutes)',
      description: 'How long before user sessions expire',
      type: 'number',
      value: 60
    },
    {
      id: 'two-factor-auth',
      category: 'Security',
      name: 'Two-Factor Authentication',
      description: 'Require 2FA for all user accounts',
      type: 'boolean',
      value: true
    },

    // Database Settings
    {
      id: 'db-connection-pool',
      category: 'Database',
      name: 'Connection Pool Size',
      description: 'Maximum number of database connections',
      type: 'number',
      value: 20
    },
    {
      id: 'db-query-timeout',
      category: 'Database',
      name: 'Query Timeout (seconds)',
      description: 'Maximum time for database queries',
      type: 'number',
      value: 30
    },

    // UI Settings
    {
      id: 'theme',
      category: 'UI',
      name: 'Theme',
      description: 'Default theme for the control panel',
      type: 'select',
      value: 'system',
      options: ['light', 'dark', 'system']
    },
    {
      id: 'language',
      category: 'UI',
      name: 'Language',
      description: 'Interface language',
      type: 'select',
      value: 'en',
      options: ['en', 'es', 'fr', 'de', 'zh']
    },
    {
      id: 'timezone',
      category: 'UI',
      name: 'Timezone',
      description: 'Default timezone for dates and times',
      type: 'select',
      value: 'UTC',
      options: ['UTC', 'EST', 'PST', 'CET', 'JST']
    }
  ]

  // Initialize settings on component mount
  useEffect(() => {
    if (settings.length === 0) {
      setSettings(defaultSettings)
      setLoading(false)
    }
  }, [settings.length])

  const [activeCategory, setActiveCategory] = useState('System')
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false)

  const categories = ['System', 'API', 'AI', 'Security', 'Database', 'UI']

  const filteredSettings = settings.filter(setting => setting.category === activeCategory)

  const handleSettingChange = (id: string, value: any) => {
    setSettings(settings.map(setting =>
      setting.id === id ? { ...setting, value } : setting
    ))
    setHasUnsavedChanges(true)
  }

  const handleSave = () => {
    // TODO: Implement save logic
    console.log('Saving settings:', settings)
    setHasUnsavedChanges(false)
  }

  const handleRefresh = () => {
    // TODO: Implement refresh logic
    console.log('Refreshing settings')
  }

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'System': return <Settings className="w-4 h-4" />
      case 'API': return <Globe className="w-4 h-4" />
      case 'AI': return <User className="w-4 h-4" />
      case 'Security': return <Shield className="w-4 h-4" />
      case 'Database': return <Database className="w-4 h-4" />
      case 'UI': return <Palette className="w-4 h-4" />
      default: return <Settings className="w-4 h-4" />
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Settings</h1>
          <p className="text-gray-600 dark:text-gray-400">Configure system settings and preferences</p>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            onClick={handleRefresh}
            disabled={saving}
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </Button>
          <Button
            onClick={handleSave}
            disabled={saving || !hasUnsavedChanges}
          >
            <Save className="w-4 h-4 mr-2" />
            {saving ? 'Saving...' : 'Save Changes'}
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Categories Sidebar */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle className="text-lg">Categories</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {categories.map((category) => (
              <button
                key={category}
                onClick={() => setActiveCategory(category)}
                className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left transition-colors ${
                  activeCategory === category
                    ? 'bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300'
                    : 'hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-700 dark:text-gray-300'
                }`}
              >
                {getCategoryIcon(category)}
                {category}
              </button>
            ))}
          </CardContent>
        </Card>

        {/* Settings Content */}
        <Card className="lg:col-span-3">
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              {getCategoryIcon(activeCategory)}
              {activeCategory} Settings
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              {filteredSettings.map((setting) => (
                <div key={setting.id} className="border-b border-gray-200 dark:border-gray-700 pb-6 last:border-b-0">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <h3 className="font-medium text-gray-900 dark:text-white">{setting.name}</h3>
                        {setting.requiresRestart && (
                          <Badge variant="secondary" className="text-xs">
                            Requires Restart
                          </Badge>
                        )}
                      </div>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">{setting.description}</p>

                      {/* Setting Input */}
                      <div className="max-w-md">
                        {setting.type === 'text' && (
                          <input
                            type="text"
                            value={setting.value}
                            onChange={(e) => handleSettingChange(setting.id, e.target.value)}
                            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                          />
                        )}

                        {setting.type === 'number' && (
                          <input
                            type="number"
                            value={setting.value}
                            onChange={(e) => handleSettingChange(setting.id, parseFloat(e.target.value))}
                            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                          />
                        )}

                        {setting.type === 'boolean' && (
                          <label className="flex items-center gap-2">
                            <input
                              type="checkbox"
                              checked={setting.value}
                              onChange={(e) => handleSettingChange(setting.id, e.target.checked)}
                              className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 dark:focus:ring-blue-600 dark:ring-offset-gray-800 focus:ring-2 dark:bg-gray-700 dark:border-gray-600"
                            />
                            <span className="text-sm text-gray-700 dark:text-gray-300">
                              {setting.value ? 'Enabled' : 'Disabled'}
                            </span>
                          </label>
                        )}

                        {setting.type === 'select' && setting.options && (
                          <select
                            value={setting.value}
                            onChange={(e) => handleSettingChange(setting.id, e.target.value)}
                            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                          >
                            {setting.options.map((option) => (
                              <option key={option} value={option}>
                                {option.charAt(0).toUpperCase() + option.slice(1)}
                              </option>
                            ))}
                          </select>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

export default SettingsPanel