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
