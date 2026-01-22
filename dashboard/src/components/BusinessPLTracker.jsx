import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, BarChart, Bar } from 'recharts';
import { TrendingUp, TrendingDown, DollarSign, AlertTriangle, Target } from 'lucide-react';

const BusinessPLTracker = () => {
  const [plData, setPlData] = useState({
    summary: {
      totalRevenue: 125000,
      totalExpenses: 87500,
      netProfit: 37500,
      marginPercent: 30
    },
    trends: [
      { month: 'Jan', revenue: 95000, expenses: 72000, profit: 23000 },
      { month: 'Feb', revenue: 105000, expenses: 78000, profit: 27000 },
      { month: 'Mar', revenue: 115000, expenses: 82000, profit: 33000 },
      { month: 'Apr', revenue: 125000, expenses: 87500, profit: 37500 }
    ],
    revenueByWorkflow: [
      { name: 'Content Creation', value: 45000, color: '#8884d8' },
      { name: 'Data Analysis', value: 35000, color: '#82ca9d' },
      { name: 'Automation', value: 30000, color: '#ffc658' },
      { name: 'Consulting', value: 15000, color: '#ff7300' }
    ],
    expensesBreakdown: [
      { category: 'LLM Costs', amount: 45000, percent: 51.4 },
      { category: 'Infrastructure', amount: 25000, percent: 28.6 },
      { category: 'Tools & APIs', amount: 12500, percent: 14.3 },
      { category: 'Other', amount: 5000, percent: 5.7 }
    ],
    roiByWorkflow: [
      { workflow: 'Content Creation', investment: 15000, revenue: 45000, roi: 200 },
      { workflow: 'Data Analysis', investment: 12000, revenue: 35000, roi: 191 },
      { workflow: 'Automation', investment: 18000, revenue: 30000, roi: 66 },
      { workflow: 'Consulting', investment: 8000, revenue: 15000, roi: 87 }
    ]
  });

  const [selectedPeriod, setSelectedPeriod] = useState('30d');
  const [budgetAlerts, setBudgetAlerts] = useState([
    { type: 'warning', message: 'LLM costs approaching 50% of budget', threshold: 80 },
    { type: 'info', message: 'Infrastructure costs within normal range', threshold: 60 }
  ]);

  const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300'];

  return (
    <div className="business-pl-tracker">
      {/* Period Selector */}
      <div className="card glass mb-6">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold">Business P&L Tracker</h3>
          <select
            value={selectedPeriod}
            onChange={(e) => setSelectedPeriod(e.target.value)}
            className="bg-gray-800 border border-gray-600 rounded px-3 py-1 text-sm"
          >
            <option value="7d">Last 7 days</option>
            <option value="30d">Last 30 days</option>
            <option value="90d">Last 90 days</option>
            <option value="1y">Last year</option>
          </select>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
        <div className="card glass">
          <div className="flex items-center gap-3">
            <DollarSign className="w-8 h-8 text-green-500" />
            <div>
              <div className="text-2xl font-bold text-green-500">
                ${plData.summary.totalRevenue.toLocaleString()}
              </div>
              <div className="text-sm text-gray-400">Total Revenue</div>
            </div>
          </div>
        </div>

        <div className="card glass">
          <div className="flex items-center gap-3">
            <TrendingDown className="w-8 h-8 text-red-500" />
            <div>
              <div className="text-2xl font-bold text-red-500">
                ${plData.summary.totalExpenses.toLocaleString()}
              </div>
              <div className="text-sm text-gray-400">Total Expenses</div>
            </div>
          </div>
        </div>

        <div className="card glass">
          <div className="flex items-center gap-3">
            <TrendingUp className="w-8 h-8 text-green-500" />
            <div>
              <div className="text-2xl font-bold text-green-500">
                ${plData.summary.netProfit.toLocaleString()}
              </div>
              <div className="text-sm text-gray-400">Net Profit</div>
            </div>
          </div>
        </div>

        <div className="card glass">
          <div className="flex items-center gap-3">
            <Target className="w-8 h-8 text-blue-500" />
            <div>
              <div className="text-2xl font-bold text-blue-500">
                {plData.summary.marginPercent}%
              </div>
              <div className="text-sm text-gray-400">Margin %</div>
            </div>
          </div>
        </div>
      </div>

      {/* Budget Alerts */}
      {budgetAlerts.length > 0 && (
        <div className="card glass mb-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-yellow-500" />
            Budget Alerts
          </h3>
          <div className="space-y-2">
            {budgetAlerts.map((alert, index) => (
              <div key={index} className={`p-3 rounded flex items-center gap-3 ${
                alert.type === 'warning' ? 'bg-yellow-500 bg-opacity-20' : 'bg-blue-500 bg-opacity-20'
              }`}>
                <AlertTriangle className={`w-4 h-4 ${
                  alert.type === 'warning' ? 'text-yellow-500' : 'text-blue-500'
                }`} />
                <span className="text-sm">{alert.message}</span>
                <span className="text-xs text-gray-400 ml-auto">{alert.threshold}%</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Historical Trends */}
        <div className="card glass">
          <h3 className="text-lg font-semibold mb-4">Historical Trends</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={plData.trends}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month" />
              <YAxis />
              <Tooltip formatter={(value) => `$${value.toLocaleString()}`} />
              <Line type="monotone" dataKey="revenue" stroke="#22c55e" strokeWidth={2} name="Revenue" />
              <Line type="monotone" dataKey="expenses" stroke="#ef4444" strokeWidth={2} name="Expenses" />
              <Line type="monotone" dataKey="profit" stroke="#3b82f6" strokeWidth={2} name="Profit" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Revenue by Workflow Type */}
        <div className="card glass">
          <h3 className="text-lg font-semibold mb-4">Revenue by Workflow Type</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={plData.revenueByWorkflow}
                cx="50%"
                cy="50%"
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
              >
                {plData.revenueByWorkflow.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip formatter={(value) => `$${value.toLocaleString()}`} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Expense Breakdown */}
        <div className="card glass">
          <h3 className="text-lg font-semibold mb-4">Expense Breakdown</h3>
          <div className="space-y-4">
            {plData.expensesBreakdown.map((expense, index) => (
              <div key={index} className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div
                    className="w-4 h-4 rounded"
                    style={{ backgroundColor: COLORS[index % COLORS.length] }}
                  />
                  <span className="text-sm">{expense.category}</span>
                </div>
                <div className="text-right">
                  <div className="font-semibold">${expense.amount.toLocaleString()}</div>
                  <div className="text-xs text-gray-400">{expense.percent}%</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* ROI by Workflow */}
        <div className="card glass">
          <h3 className="text-lg font-semibold mb-4">ROI by Workflow</h3>
          <div className="space-y-4">
            {plData.roiByWorkflow.map((workflow, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-800 rounded">
                <div>
                  <div className="font-medium">{workflow.workflow}</div>
                  <div className="text-xs text-gray-400">
                    Investment: ${workflow.investment.toLocaleString()}
                  </div>
                </div>
                <div className="text-right">
                  <div className="font-semibold text-green-500">{workflow.roi}% ROI</div>
                  <div className="text-xs text-gray-400">
                    Revenue: ${workflow.revenue.toLocaleString()}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Cost Attribution Table */}
      <div className="card glass">
        <h3 className="text-lg font-semibold mb-4">Cost Attribution per Workflow</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-700">
                <th className="text-left py-2">Workflow Type</th>
                <th className="text-right py-2">Revenue</th>
                <th className="text-right py-2">LLM Costs</th>
                <th className="text-right py-2">Infra Costs</th>
                <th className="text-right py-2">Tool Costs</th>
                <th className="text-right py-2">Net Profit</th>
                <th className="text-right py-2">Margin %</th>
              </tr>
            </thead>
            <tbody>
              {plData.revenueByWorkflow.map((workflow, index) => {
                const expenses = plData.expensesBreakdown;
                const llmCost = Math.round(workflow.value * 0.4);
                const infraCost = Math.round(workflow.value * 0.2);
                const toolCost = Math.round(workflow.value * 0.1);
                const totalCost = llmCost + infraCost + toolCost;
                const profit = workflow.value - totalCost;
                const margin = ((profit / workflow.value) * 100).toFixed(1);

                return (
                  <tr key={index} className="border-b border-gray-800">
                    <td className="py-2">{workflow.name}</td>
                    <td className="text-right py-2">${workflow.value.toLocaleString()}</td>
                    <td className="text-right py-2">${llmCost.toLocaleString()}</td>
                    <td className="text-right py-2">${infraCost.toLocaleString()}</td>
                    <td className="text-right py-2">${toolCost.toLocaleString()}</td>
                    <td className="text-right py-2 text-green-500">${profit.toLocaleString()}</td>
                    <td className="text-right py-2">{margin}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default BusinessPLTracker;