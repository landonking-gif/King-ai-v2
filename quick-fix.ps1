#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Quick fix script for common King AI v3 issues
#>

$SERVER_IP = "3.236.144.91"
$SSH_KEY = "king-ai-v3/agentic-framework-main/king-ai-studio.pem"

Write-Host "`n🔧 KING AI v3 QUICK FIX`n" -ForegroundColor Cyan

Write-Host "Select an option:" -ForegroundColor Yellow
Write-Host "1. Restart all services" -ForegroundColor White
Write-Host "2. Restart Ollama (LLM)" -ForegroundColor White
Write-Host "3. Restart Orchestrator only" -ForegroundColor White
Write-Host "4. Check and pull Ollama model" -ForegroundColor White
Write-Host "5. Full system reset (down + up)" -ForegroundColor White
Write-Host "6. View live logs" -ForegroundColor White
Write-Host "7. Check disk space" -ForegroundColor White
Write-Host "8. Test AI chat endpoint" -ForegroundColor White
Write-Host "0. Exit" -ForegroundColor White

$choice = Read-Host "`nEnter choice (0-8)"

switch ($choice) {
    "1" {
        Write-Host "`n📦 Restarting all services..." -ForegroundColor Yellow
        ssh -i $SSH_KEY ubuntu@$SERVER_IP "cd king-ai-v3/agentic-framework-main && docker-compose restart"
        Write-Host "✅ Services restarted. Wait 30 seconds then test." -ForegroundColor Green
    }
    
    "2" {
        Write-Host "`n🤖 Restarting Ollama..." -ForegroundColor Yellow
        ssh -i $SSH_KEY ubuntu@$SERVER_IP "docker restart ollama"
        Start-Sleep -Seconds 5
        ssh -i $SSH_KEY ubuntu@$SERVER_IP "docker exec ollama ollama list"
        Write-Host "✅ Ollama restarted" -ForegroundColor Green
    }
    
    "3" {
        Write-Host "`n🎯 Restarting Orchestrator..." -ForegroundColor Yellow
        ssh -i $SSH_KEY ubuntu@$SERVER_IP "docker restart orchestrator"
        Write-Host "✅ Orchestrator restarted" -ForegroundColor Green
    }
    
    "4" {
        Write-Host "`n📥 Checking Ollama model..." -ForegroundColor Yellow
        $models = ssh -i $SSH_KEY ubuntu@$SERVER_IP "docker exec ollama ollama list"
        Write-Host $models
        
        if ($models -notmatch "llama3.1:70b") {
            Write-Host "`n⚠️  Model not found. Pulling llama3.1:70b (this takes 10-15 minutes)..." -ForegroundColor Yellow
            ssh -i $SSH_KEY ubuntu@$SERVER_IP "docker exec ollama ollama pull llama3.1:70b"
            Write-Host "✅ Model pulled successfully" -ForegroundColor Green
        } else {
            Write-Host "✅ Model llama3.1:70b is available" -ForegroundColor Green
        }
    }
    
    "5" {
        Write-Host "`n🔄 Full system reset..." -ForegroundColor Yellow
        Write-Host "This will stop and restart all services." -ForegroundColor Red
        $confirm = Read-Host "Continue? (yes/no)"
        
        if ($confirm -eq "yes") {
            ssh -i $SSH_KEY ubuntu@$SERVER_IP "cd king-ai-v3/agentic-framework-main && docker-compose down"
            Start-Sleep -Seconds 5
            ssh -i $SSH_KEY ubuntu@$SERVER_IP "cd king-ai-v3/agentic-framework-main && docker-compose up -d"
            Write-Host "✅ System reset complete. Wait 60 seconds for all services to start." -ForegroundColor Green
        }
    }
    
    "6" {
        Write-Host "`n📋 Live logs (Ctrl+C to stop)..." -ForegroundColor Yellow
        Write-Host "Which service?" -ForegroundColor White
        Write-Host "1. Orchestrator" -ForegroundColor White
        Write-Host "2. Ollama" -ForegroundColor White
        Write-Host "3. Memory Service" -ForegroundColor White
        Write-Host "4. All services" -ForegroundColor White
        
        $logChoice = Read-Host "Enter choice"
        
        switch ($logChoice) {
            "1" { ssh -i $SSH_KEY ubuntu@$SERVER_IP "docker logs -f orchestrator" }
            "2" { ssh -i $SSH_KEY ubuntu@$SERVER_IP "docker logs -f ollama" }
            "3" { ssh -i $SSH_KEY ubuntu@$SERVER_IP "docker logs -f memory-service" }
            "4" { ssh -i $SSH_KEY ubuntu@$SERVER_IP "docker-compose logs -f" }
        }
    }
    
    "7" {
        Write-Host "`n💾 Checking disk space..." -ForegroundColor Yellow
        ssh -i $SSH_KEY ubuntu@$SERVER_IP "df -h | grep -E '^Filesystem|/$'"
        ssh -i $SSH_KEY ubuntu@$SERVER_IP "docker system df"
        
        Write-Host "`nClean up Docker? (yes/no)" -ForegroundColor Yellow
        $cleanup = Read-Host
        if ($cleanup -eq "yes") {
            ssh -i $SSH_KEY ubuntu@$SERVER_IP "docker system prune -f"
            Write-Host "✅ Cleanup complete" -ForegroundColor Green
        }
    }
    
    "8" {
        Write-Host "`n💬 Testing AI chat endpoint..." -ForegroundColor Yellow
        
        $body = @{
            text = "Hello, are you working?"
            session_id = "test-$(Get-Date -Format 'yyyyMMddHHmmss')"
        } | ConvertTo-Json
        
        try {
            $response = Invoke-RestMethod -Uri "http://${SERVER_IP}:8000/api/chat/message" `
                -Method Post `
                -Body $body `
                -ContentType "application/json" `
                -TimeoutSec 30
            
            Write-Host "✅ AI Response:" -ForegroundColor Green
            Write-Host $response.response -ForegroundColor White
        } catch {
            Write-Host "❌ Chat endpoint failed:" -ForegroundColor Red
            Write-Host $_.Exception.Message -ForegroundColor Red
            
            Write-Host "`nTroubleshooting steps:" -ForegroundColor Yellow
            Write-Host "1. Check if Ollama has the model:" -ForegroundColor White
            ssh -i $SSH_KEY ubuntu@$SERVER_IP "docker exec ollama ollama list"
            
            Write-Host "`n2. Check Orchestrator logs:" -ForegroundColor White
            ssh -i $SSH_KEY ubuntu@$SERVER_IP "docker logs orchestrator --tail 50"
        }
    }
    
    "0" {
        Write-Host "`nExiting..." -ForegroundColor Gray
        exit
    }
    
    default {
        Write-Host "`n❌ Invalid choice" -ForegroundColor Red
    }
}

Write-Host "`n✅ Done!`n" -ForegroundColor Green
