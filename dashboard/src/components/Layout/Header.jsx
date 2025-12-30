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
