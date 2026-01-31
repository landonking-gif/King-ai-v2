#!/usr/bin/env pwsh

param(
    [Parameter(Mandatory=$true)]
    [string]$IpAddress
)

Write-Host "Testing deployment to $IpAddress"

# Simple test functions
function Test-Function {
    Write-Host "Function works"
}

Test-Function
Write-Host "Script loaded successfully"