import React, { useState, useEffect } from 'react';
import { Card } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface PLSummary {
  period: string;
  total_revenue: number;
  total_expenses: number;
  net_profit: number;
  margin_percent: number;
  transaction_count: number;
  avg_transaction_value: number;
}

interface Trend {
  date: string;
  revenue: number;
  expenses: number;
  profit: number;
  margin_percent: number;
}

interface CostBreakdown {
  category: string;
  amount: number;
  percentage: number;
  transactions: number;
}

export const PLDashboard: React.FC = () => {
  const [summary, setSummary] = useState<PLSummary | null>(null);
  const [trends, setTrends] = useState<Trend[]>([]);
  const [breakdown, setBreakdown] = useState<CostBreakdown[]>([]);
  const [loading, setLoading] = useState(false);
  const [period, setPeriod] = useState<'daily' | 'monthly' | 'yearly'>('monthly');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchPLData();
    const interval = setInterval(fetchPLData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, [period]);

  const fetchPLData = async () => {
    setLoading(true);
    try {
      const token = localStorage.getItem('token');
      const headers = {
        'Authorization': `Bearer ${token}`,
      };

      // Fetch summary
      const summaryRes = await fetch(`/api/business/pl/summary?period=${period}`, { headers });
      if (summaryRes.ok) {
        setSummary(await summaryRes.json());
      }

      // Fetch trends
      const trendsRes = await fetch('/api/business/pl/trends?days=30', { headers });
      if (trendsRes.ok) {
        const data = await trendsRes.json();
        setTrends(data.trends || []);
      }

      // Fetch breakdown
      const breakdownRes = await fetch('/api/business/pl/breakdown', { headers });
      if (breakdownRes.ok) {
        const data = await breakdownRes.json();
        setBreakdown(data.breakdown || []);
      }

      setError(null);
    } catch (err) {
      setError('Failed to fetch P&L data: ' + String(err));
    } finally {
      setLoading(false);
    }
  };

  const getProfitColor = (profit: number): string => {
    if (profit > 0) return 'text-green-600';
    if (profit < 0) return 'text-red-600';
    return 'text-gray-600';
  };

  return (
    <div className="p-6 bg-white">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Business P&L Tracker</h1>
          <p className="text-gray-600 mt-2">Revenue, expenses, and profitability tracking</p>
        </div>
        <div className="flex gap-2">
          <Button
            variant={period === 'daily' ? 'default' : 'outline'}
            onClick={() => setPeriod('daily')}
          >
            Daily
          </Button>
          <Button
            variant={period === 'monthly' ? 'default' : 'outline'}
            onClick={() => setPeriod('monthly')}
          >
            Monthly
          </Button>
          <Button
            variant={period === 'yearly' ? 'default' : 'outline'}
            onClick={() => setPeriod('yearly')}
          >
            Yearly
          </Button>
        </div>
      </div>

      {error && (
        <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
          {error}
        </div>
      )}

      {loading && !summary && (
        <div className="text-center p-8">
          <p className="text-gray-500">Loading P&L data...</p>
        </div>
      )}

      {summary && (
        <>
          {/* Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 mb-6">
            <Card className="p-4">
              <p className="text-sm text-gray-600 mb-1">Total Revenue</p>
              <p className="text-2xl font-bold text-green-600">${summary.total_revenue.toFixed(2)}</p>
              <p className="text-xs text-gray-500 mt-2">
                {summary.transaction_count} transactions
              </p>
            </Card>
            <Card className="p-4">
              <p className="text-sm text-gray-600 mb-1">Total Expenses</p>
              <p className="text-2xl font-bold text-red-600">${summary.total_expenses.toFixed(2)}</p>
              <p className="text-xs text-gray-500 mt-2">
                {(summary.total_expenses / summary.total_revenue * 100).toFixed(1)}% of revenue
              </p>
            </Card>
            <Card className="p-4">
              <p className="text-sm text-gray-600 mb-1">Net Profit</p>
              <p className={`text-2xl font-bold ${getProfitColor(summary.net_profit)}`}>
                ${summary.net_profit.toFixed(2)}
              </p>
              <Badge className={summary.net_profit > 0 ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}>
                {summary.margin_percent.toFixed(1)}% margin
              </Badge>
            </Card>
            <Card className="p-4">
              <p className="text-sm text-gray-600 mb-1">Avg Transaction</p>
              <p className="text-2xl font-bold text-blue-600">${summary.avg_transaction_value.toFixed(2)}</p>
              <p className="text-xs text-gray-500 mt-2">
                Per transaction
              </p>
            </Card>
            <Card className="p-4">
              <p className="text-sm text-gray-600 mb-1">Profit Margin</p>
              <p className="text-2xl font-bold text-purple-600">{summary.margin_percent.toFixed(1)}%</p>
              <p className="text-xs text-gray-500 mt-2">
                Period: {summary.period}
              </p>
            </Card>
          </div>

          {/* Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
            {/* Trend Chart */}
            <Card className="p-4">
              <h2 className="text-lg font-semibold mb-4">Revenue vs Expenses Trend</h2>
              {trends.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={trends}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip formatter={(value) => `$${Number(value).toFixed(2)}`} />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="revenue"
                      stroke="#22c55e"
                      name="Revenue"
                      strokeWidth={2}
                    />
                    <Line
                      type="monotone"
                      dataKey="expenses"
                      stroke="#ef4444"
                      name="Expenses"
                      strokeWidth={2}
                    />
                    <Line
                      type="monotone"
                      dataKey="profit"
                      stroke="#3b82f6"
                      name="Profit"
                      strokeWidth={2}
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-gray-500">No trend data available</p>
              )}
            </Card>

            {/* Cost Breakdown Chart */}
            <Card className="p-4">
              <h2 className="text-lg font-semibold mb-4">Cost Breakdown by Category</h2>
              {breakdown.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={breakdown}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="category" angle={-45} textAnchor="end" height={80} />
                    <YAxis />
                    <Tooltip formatter={(value) => `$${Number(value).toFixed(2)}`} />
                    <Bar dataKey="amount" fill="#8b5cf6" name="Amount" />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-gray-500">No cost data available</p>
              )}
            </Card>
          </div>

          {/* Cost Breakdown Table */}
          <Card className="p-4">
            <h2 className="text-lg font-semibold mb-4">Detailed Cost Breakdown</h2>
            {breakdown.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="border-b-2 border-gray-200">
                    <tr>
                      <th className="text-left p-2">Category</th>
                      <th className="text-right p-2">Amount</th>
                      <th className="text-right p-2">% of Total</th>
                      <th className="text-right p-2">Transactions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {breakdown.map((item, idx) => (
                      <tr key={idx} className="border-b border-gray-100 hover:bg-gray-50">
                        <td className="p-2">{item.category}</td>
                        <td className="text-right p-2 font-semibold">${item.amount.toFixed(2)}</td>
                        <td className="text-right p-2">
                          <Badge variant="secondary">{item.percentage.toFixed(1)}%</Badge>
                        </td>
                        <td className="text-right p-2 text-gray-600">{item.transactions}</td>
                      </tr>
                    ))}
                  </tbody>
                  <tfoot className="border-t-2 border-gray-200 font-semibold">
                    <tr>
                      <td className="p-2">Total Expenses</td>
                      <td className="text-right p-2">
                        ${breakdown.reduce((sum, item) => sum + item.amount, 0).toFixed(2)}
                      </td>
                      <td className="text-right p-2">100%</td>
                      <td className="text-right p-2">
                        {breakdown.reduce((sum, item) => sum + item.transactions, 0)}
                      </td>
                    </tr>
                  </tfoot>
                </table>
              </div>
            ) : (
              <p className="text-gray-500">No cost breakdown data available</p>
            )}
          </Card>
        </>
      )}
    </div>
  );
};

export default PLDashboard;
