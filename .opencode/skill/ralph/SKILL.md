---
name: ralph
description: Convert markdown PRDs to JSON format for automated processing.
---

# Ralph Conversion Skill

When given a markdown PRD file, convert it to JSON format with the following structure:

{
  "title": "string",
  "overview": "string",
  "goals": ["array of strings"],
  "targetAudience": "string",
  "userStories": [
    {
      "id": "unique id",
      "title": "string",
      "description": "string",
      "acceptanceCriteria": ["array of strings"]
    }
  ],
  "technicalRequirements": ["array"],
  "timeline": "string",
  "successMetrics": ["array"]
}

Parse the markdown sections accordingly.