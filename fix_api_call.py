#!/usr/bin/env python3

with open('/home/ubuntu/dashboard/src/components/TalkToKingAI.jsx', 'r') as f:
    content = f.read()

# Replace the API call to include required fields
old_api_call = """body: JSON.stringify({ 
          text: userMessage.content
        })"""

new_api_call = """body: JSON.stringify({ 
          text: userMessage.content,
          user_id: 'dashboard-user',
          business_id: 'default-business',
          agent_id: 'primary'
        })"""

content = content.replace(old_api_call, new_api_call)

# Also try alternative format
old_api_call2 = """body: JSON.stringify({ text: userMessage.content })"""
new_api_call2 = """body: JSON.stringify({ 
          text: userMessage.content,
          user_id: 'dashboard-user',
          business_id: 'default-business',
          agent_id: 'primary'
        })"""

content = content.replace(old_api_call2, new_api_call2)

with open('/home/ubuntu/dashboard/src/components/TalkToKingAI.jsx', 'w') as f:
    f.write(content)

print('Fixed TalkToKingAI.jsx API call')
