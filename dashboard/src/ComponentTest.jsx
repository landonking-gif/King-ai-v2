import React from 'react';
import {
  BusinessCard,
  BusinessList,
  RevenueChart,
  MetricCard,
  DonutChart,
  HealthBadge,
  ProgressBar,
  DataTable,
  StatusIndicator,
} from './components';

// Sample data for testing components
const sampleBusinesses = [
  {
    id: 1,
    name: 'Tech Startup',
    type: 'SaaS',
    stage: 'growth',
    health_score: 85,
    revenue: 150000,
    growth_rate: 25,
    active_tasks: 5,
  },
  {
    id: 2,
    name: 'E-commerce Store',
    type: 'Retail',
    stage: 'scaling',
    health_score: 72,
    revenue: 250000,
    growth_rate: 15,
    active_tasks: 3,
  },
  {
    id: 3,
    name: 'Consulting Firm',
    type: 'Service',
    stage: 'maturity',
    health_score: 90,
    revenue: 500000,
    growth_rate: 10,
    active_tasks: 8,
  },
];

const revenueData = [
  { label: 'Jan', value: 10000 },
  { label: 'Feb', value: 15000 },
  { label: 'Mar', value: 22000 },
  { label: 'Apr', value: 28000 },
  { label: 'May', value: 35000 },
  { label: 'Jun', value: 42000 },
];

const donutData = [
  { label: 'SaaS', value: 40, color: '#6366f1' },
  { label: 'Retail', value: 30, color: '#10b981' },
  { label: 'Service', value: 20, color: '#f59e0b' },
  { label: 'Other', value: 10, color: '#ef4444' },
];

const tableColumns = [
  { key: 'name', label: 'Business Name' },
  { key: 'type', label: 'Type' },
  { key: 'revenue', label: 'Revenue', render: (val) => `$${val?.toLocaleString()}` },
  { key: 'health_score', label: 'Health', render: (val) => <HealthBadge score={val} showLabel={false} /> },
];

function ComponentTest() {
  return (
    <div style={{ padding: '40px', background: '#0f0c29', minHeight: '100vh' }}>
      <h1 style={{ color: '#fff', marginBottom: '40px' }}>Component Test Page</h1>
      
      {/* Metric Cards */}
      <section style={{ marginBottom: '40px' }}>
        <h2 style={{ color: '#fff', marginBottom: '20px' }}>Metric Cards</h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '20px' }}>
          <MetricCard
            title="Total Revenue"
            value="$900K"
            change={15}
            changeLabel="vs last month"
            icon="💰"
            trend="up"
          />
          <MetricCard
            title="Active Businesses"
            value="12"
            change={8}
            changeLabel="vs last month"
            icon="🏢"
            trend="up"
          />
          <MetricCard
            title="Avg Health Score"
            value="82"
            change={-3}
            changeLabel="vs last month"
            icon="❤️"
            trend="down"
          />
        </div>
      </section>

      {/* Charts */}
      <section style={{ marginBottom: '40px' }}>
        <h2 style={{ color: '#fff', marginBottom: '20px' }}>Charts</h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '20px' }}>
          <div style={{ background: 'rgba(20,20,30,0.8)', padding: '20px', borderRadius: '12px' }}>
            <h3 style={{ color: '#fff', marginBottom: '16px' }}>Revenue Trend</h3>
            <RevenueChart data={revenueData} height={200} />
          </div>
          <div style={{ background: 'rgba(20,20,30,0.8)', padding: '20px', borderRadius: '12px' }}>
            <h3 style={{ color: '#fff', marginBottom: '16px' }}>Business Distribution</h3>
            <DonutChart data={donutData} size={180} thickness={25} />
          </div>
        </div>
      </section>

      {/* Common Components */}
      <section style={{ marginBottom: '40px' }}>
        <h2 style={{ color: '#fff', marginBottom: '20px' }}>Common Components</h2>
        <div style={{ background: 'rgba(20,20,30,0.8)', padding: '20px', borderRadius: '12px' }}>
          <div style={{ marginBottom: '20px' }}>
            <h4 style={{ color: '#fff', marginBottom: '10px' }}>Health Badges</h4>
            <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
              <HealthBadge score={90} />
              <HealthBadge score={75} />
              <HealthBadge score={55} />
              <HealthBadge score={30} />
            </div>
          </div>
          
          <div style={{ marginBottom: '20px' }}>
            <h4 style={{ color: '#fff', marginBottom: '10px' }}>Progress Bars</h4>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <ProgressBar value={85} max={100} label="Completion" />
              <ProgressBar value={60} max={100} label="Health" />
              <ProgressBar value={30} max={100} label="Growth" />
            </div>
          </div>
          
          <div>
            <h4 style={{ color: '#fff', marginBottom: '10px' }}>Status Indicators</h4>
            <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
              <StatusIndicator status="running" />
              <StatusIndicator status="completed" />
              <StatusIndicator status="failed" />
              <StatusIndicator status="pending" />
              <StatusIndicator status="active" />
            </div>
          </div>
        </div>
      </section>

      {/* Business Cards */}
      <section style={{ marginBottom: '40px' }}>
        <h2 style={{ color: '#fff', marginBottom: '20px' }}>Business Cards</h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px' }}>
          {sampleBusinesses.map(business => (
            <BusinessCard
              key={business.id}
              business={business}
              onClick={(id) => console.log('Clicked business:', id)}
            />
          ))}
        </div>
      </section>

      {/* Business List */}
      <section style={{ marginBottom: '40px' }}>
        <h2 style={{ color: '#fff', marginBottom: '20px' }}>Business List (with filters)</h2>
        <BusinessList
          businesses={sampleBusinesses}
          onSelect={(id) => console.log('Selected business:', id)}
        />
      </section>

      {/* Data Table */}
      <section>
        <h2 style={{ color: '#fff', marginBottom: '20px' }}>Data Table</h2>
        <div style={{ background: 'rgba(20,20,30,0.8)', padding: '20px', borderRadius: '12px' }}>
          <DataTable
            columns={tableColumns}
            data={sampleBusinesses}
            onRowClick={(row) => console.log('Clicked row:', row)}
            pageSize={5}
          />
        </div>
      </section>
    </div>
  );
}

export default ComponentTest;
