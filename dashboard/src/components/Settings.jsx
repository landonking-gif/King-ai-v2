import React, { useState, useEffect } from 'react';
import './Settings.css';

const Settings = () => {
  const [settings, setSettings] = useState({
    maxAutoApproveAmount: 1000,
    approvalExpiryHours: 24,
    requireApprovalLegal: true,
    systemTimeout: 300,
    maxConcurrentAgents: 10,
    logLevel: 'info',
    enableWebSocket: true,
    databaseUrl: 'postgresql://localhost:5432/kingai',
    redisUrl: 'redis://localhost:6379'
  });

  const [activeTab, setActiveTab] = useState('general');

  const handleSettingChange = (key, value) => {
    setSettings(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const saveSettings = async () => {
    try {
      // Save settings to backend
      const response = await fetch('/api/settings', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(settings)
      });

      if (response.ok) {
        alert('Settings saved successfully!');
      } else {
        throw new Error('Failed to save settings');
      }
    } catch (error) {
      console.error('Error saving settings:', error);
      alert('Failed to save settings');
    }
  };

  const tabs = [
    { id: 'general', name: 'General', icon: '⚙️' },
    { id: 'approvals', name: 'Approvals', icon: '✅' },
    { id: 'system', name: 'System', icon: '🖥️' },
    { id: 'database', name: 'Database', icon: '🗄️' }
  ];

  return (
    <div className="settings-dashboard">
      <div className="settings-header">
        <h2>Settings & Configuration</h2>
        <p>Manage system configuration and preferences</p>
      </div>

      <div className="settings-container">
        <div className="settings-tabs">
          {tabs.map(tab => (
            <button
              key={tab.id}
              className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              <span className="tab-icon">{tab.icon}</span>
              {tab.name}
            </button>
          ))}
        </div>

        <div className="settings-content">
          {activeTab === 'general' && (
            <div className="settings-section">
              <h3>General Settings</h3>
              <div className="setting-group">
                <label>
                  <span>System Timeout (seconds)</span>
                  <input
                    type="number"
                    value={settings.systemTimeout}
                    onChange={(e) => handleSettingChange('systemTimeout', parseInt(e.target.value))}
                  />
                </label>
                <label>
                  <span>Max Concurrent Agents</span>
                  <input
                    type="number"
                    value={settings.maxConcurrentAgents}
                    onChange={(e) => handleSettingChange('maxConcurrentAgents', parseInt(e.target.value))}
                  />
                </label>
                <label>
                  <span>Log Level</span>
                  <select
                    value={settings.logLevel}
                    onChange={(e) => handleSettingChange('logLevel', e.target.value)}
                  >
                    <option value="debug">Debug</option>
                    <option value="info">Info</option>
                    <option value="warning">Warning</option>
                    <option value="error">Error</option>
                  </select>
                </label>
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={settings.enableWebSocket}
                    onChange={(e) => handleSettingChange('enableWebSocket', e.target.checked)}
                  />
                  <span>Enable WebSocket Connections</span>
                </label>
              </div>
            </div>
          )}

          {activeTab === 'approvals' && (
            <div className="settings-section">
              <h3>Approval Settings</h3>
              <div className="setting-group">
                <label>
                  <span>Max Auto-Approve Amount ($)</span>
                  <input
                    type="number"
                    value={settings.maxAutoApproveAmount}
                    onChange={(e) => handleSettingChange('maxAutoApproveAmount', parseInt(e.target.value))}
                  />
                </label>
                <label>
                  <span>Approval Expiry (hours)</span>
                  <input
                    type="number"
                    value={settings.approvalExpiryHours}
                    onChange={(e) => handleSettingChange('approvalExpiryHours', parseInt(e.target.value))}
                  />
                </label>
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={settings.requireApprovalLegal}
                    onChange={(e) => handleSettingChange('requireApprovalLegal', e.target.checked)}
                  />
                  <span>Require Approval for Legal Actions</span>
                </label>
              </div>
            </div>
          )}

          {activeTab === 'system' && (
            <div className="settings-section">
              <h3>System Configuration</h3>
              <div className="setting-group">
                <div className="info-section">
                  <h4>System Information</h4>
                  <p><strong>Version:</strong> King AI v3.0.0</p>
                  <p><strong>Environment:</strong> Development</p>
                  <p><strong>Uptime:</strong> 2 days, 4 hours</p>
                </div>
                <div className="actions">
                  <button className="btn-secondary">Restart Services</button>
                  <button className="btn-secondary">Clear Cache</button>
                  <button className="btn-danger">Reset to Defaults</button>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'database' && (
            <div className="settings-section">
              <h3>Database Configuration</h3>
              <div className="setting-group">
                <label>
                  <span>Database URL</span>
                  <input
                    type="text"
                    value={settings.databaseUrl}
                    onChange={(e) => handleSettingChange('databaseUrl', e.target.value)}
                  />
                </label>
                <label>
                  <span>Redis URL</span>
                  <input
                    type="text"
                    value={settings.redisUrl}
                    onChange={(e) => handleSettingChange('redisUrl', e.target.value)}
                  />
                </label>
                <div className="actions">
                  <button className="btn-secondary">Test Connection</button>
                  <button className="btn-secondary">Backup Database</button>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="settings-actions">
          <button className="btn-primary" onClick={saveSettings}>
            Save Settings
          </button>
          <button className="btn-secondary" onClick={() => window.location.reload()}>
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
};

export default Settings;