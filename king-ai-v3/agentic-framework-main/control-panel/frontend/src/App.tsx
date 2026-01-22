import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Toaster } from 'sonner'
import Dashboard from './pages/Dashboard'
import WorkflowStudioPage from './pages/WorkflowStudio'
import ApprovalCenterPage from './pages/ApprovalCenter'
import AnalyticsPage from './pages/Analytics'
import SettingsPage from './pages/Settings'
import AgentControlCenterPage from './pages/AgentControlCenter'
import PLTrackerPage from './pages/PLTracker'
import ConversationalInterfacePage from './pages/ConversationalInterface'
import Login from './components/Login'
import ProtectedRoute from './components/ProtectedRoute'
import Layout from './components/Layout'
import './App.css'

const queryClient = new QueryClient()

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <div className="App">
          <Routes>
            <Route path="/login" element={<Login />} />
            <Route path="/*" element={
              <ProtectedRoute>
                <Layout>
                  <Routes>
                    <Route path="/" element={<Dashboard />} />
                    <Route path="/workflow-studio" element={<WorkflowStudioPage />} />
                    <Route path="/approvals" element={<ApprovalCenterPage />} />
                    <Route path="/agents" element={<AgentControlCenterPage />} />
                    <Route path="/pl-tracker" element={<PLTrackerPage />} />
                    <Route path="/talk-to-ai" element={<ConversationalInterfacePage />} />
                    <Route path="/analytics" element={<AnalyticsPage />} />
                    <Route path="/settings" element={<SettingsPage />} />
                  </Routes>
                </Layout>
              </ProtectedRoute>
            } />
          </Routes>
        </div>
      </Router>
      <Toaster />
    </QueryClientProvider>
  )
}

export default App