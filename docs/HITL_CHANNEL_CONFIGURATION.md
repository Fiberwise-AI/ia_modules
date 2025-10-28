# HITL Channel Configuration Guide

This guide shows how to configure notification channels and user assignments for Human-in-the-Loop interactions.

## Available Channels

- `web` - Frontend UI (always works, default)
- `email` - Email notifications (placeholder, ready for implementation)
- `slack` - Slack messages (placeholder, ready for implementation)
- `discord` - Discord messages (placeholder, ready for implementation)
- `sms` - SMS notifications (placeholder, ready for implementation)
- `teams` - Microsoft Teams (placeholder, ready for implementation)

## Configuration Methods

### Method 1: Pipeline JSON Config (Static)

**Best for**: Fixed workflows with known approvers

```json
{
  "steps": [
    {
      "id": "approval_step",
      "name": "Manager Approval",
      "step_class": "ApprovalStep",
      "module": "steps.approval",
      "config": {
        "prompt": "Please approve or reject this request",
        "timeout_seconds": 7200,
        "channels": ["web", "email", "slack"],
        "assigned_users": [
          "manager@company.com",
          "backup-manager@company.com"
        ]
      }
    }
  ]
}
```

### Method 2: Dynamic in Step Code

**Best for**: Dynamic user assignment based on data

```python
from ia_modules.pipeline.core import Step
from typing import Dict, Any

class DynamicApprovalStep(Step):
    async def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Get approvers dynamically based on amount
        amount = data.get('amount', 0)

        if amount > 10000:
            approvers = ["cfo@company.com", "ceo@company.com"]
            channels = ["web", "email", "sms"]  # Urgent - use all channels
        elif amount > 1000:
            approvers = ["manager@company.com"]
            channels = ["web", "email"]
        else:
            approvers = ["supervisor@company.com"]
            channels = ["web"]  # Low priority - web only

        return {
            "status": "human_input_required",
            "prompt": f"Approve ${amount} expense?",
            "ui_schema": {...},
            "channels": channels,
            "assigned_users": approvers,
            "timeout_seconds": 3600
        }
```

### Method 3: Hybrid (Config + Override)

**Best for**: Default settings with runtime overrides

```python
class HybridApprovalStep(Step):
    async def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Start with config defaults
        channels = self.config.get('channels', ['web'])
        users = self.config.get('assigned_users', [])

        # Override based on priority
        if data.get('priority') == 'urgent':
            channels = ["web", "email", "slack", "sms"]  # All channels
            users.append("on-call@company.com")  # Add on-call

        return {
            "status": "human_input_required",
            "channels": channels,
            "assigned_users": users,
            ...
        }
```

## Channel-Specific Configuration

### Email Channel

```json
{
  "config": {
    "channels": ["email"],
    "email_config": {
      "subject": "Approval Required: {pipeline_name}",
      "template": "approval_request",
      "cc": ["compliance@company.com"],
      "include_context": true
    }
  }
}
```

### Slack Channel

```json
{
  "config": {
    "channels": ["slack"],
    "slack_config": {
      "channel": "#approvals",
      "mention_users": true,
      "thread_replies": true,
      "include_buttons": ["Approve", "Reject", "View Details"]
    }
  }
}
```

### Discord Channel

```json
{
  "config": {
    "channels": ["discord"],
    "discord_config": {
      "guild_id": "123456789",
      "channel_id": "987654321",
      "mention_role": "@approvers",
      "embed": true
    }
  }
}
```

## User Assignment Patterns

### 1. Direct Assignment
```python
"assigned_users": ["user1@company.com", "user2@company.com"]
```

### 2. Role-Based (Future)
```python
"assigned_roles": ["manager", "admin"]  # Will look up users with these roles
```

### 3. Dynamic Lookup
```python
# In step code
async def run(self, data):
    # Look up approvers from database/API
    project_id = data.get('project_id')
    approvers = await self.get_project_approvers(project_id)

    return {
        "status": "human_input_required",
        "assigned_users": approvers,
        ...
    }
```

### 4. Escalation Chain
```python
{
  "config": {
    "assigned_users": ["supervisor@company.com"],
    "escalation": {
      "timeout_minutes": 30,
      "escalate_to": ["manager@company.com"],
      "channels": ["web", "email", "sms"]  # Escalation uses more urgent channels
    }
  }
}
```

## Querying Interactions by User

Frontend can query user-specific interactions:

```javascript
// Get all pending interactions assigned to current user
const response = await fetch('/api/hitl/pending?user_id=alice@company.com');
const myInteractions = await response.json();

// Show notification count
const count = myInteractions.length;
showNotificationBadge(count);
```

## Implementation Status

### ✅ Implemented
- Channel configuration in pipeline JSON
- Dynamic channel selection in step code
- User assignment creation in database
- User-filtered queries (`GET /api/hitl/pending?user_id=X`)
- Web channel (frontend UI)

### 📝 Placeholder (Ready for Implementation)
- Email notifier
- Slack notifier
- Discord notifier
- SMS notifier
- Teams notifier

## Adding a New Channel Notifier

To implement a new channel (e.g., email):

### 1. Create Notifier Class

```python
# ia_modules/pipeline/channels/email_notifier.py

class EmailNotifier:
    def __init__(self, smtp_config):
        self.smtp = smtp_config

    async def notify(
        self,
        interaction_id: str,
        assigned_users: List[str],
        prompt: str,
        ui_schema: Dict[str, Any],
        context_data: Dict[str, Any]
    ):
        """Send email to assigned users"""
        for user_email in assigned_users:
            await self.send_email(
                to=user_email,
                subject=f"Action Required: {prompt}",
                body=self.render_template(ui_schema, context_data),
                action_link=f"{self.base_url}/respond/{interaction_id}"
            )
```

### 2. Register in HITLManager

```python
# In HITLManager.__init__
from ia_modules.pipeline.channels.email_notifier import EmailNotifier

self.notifiers = {
    'web': None,  # Handled by frontend polling
    'email': EmailNotifier(smtp_config),
    'slack': SlackNotifier(slack_token),
    'discord': DiscordNotifier(discord_webhook),
}
```

### 3. Update notify_channels Method

```python
async def notify_channels(self, interaction_id, channels, prompt, ui_schema):
    for channel in channels:
        notifier = self.notifiers.get(channel)
        if notifier:
            await notifier.notify(interaction_id, assigned_users, prompt, ui_schema)
        else:
            logger.warning(f"No notifier configured for channel: {channel}")
```

## Complete Example Pipeline

```json
{
  "name": "Multi-Stage Approval with Escalation",
  "steps": [
    {
      "id": "initial_review",
      "name": "Initial Review",
      "step_class": "ReviewStep",
      "config": {
        "prompt": "Initial data review required",
        "channels": ["web", "email"],
        "assigned_users": ["data-analyst@company.com"],
        "timeout_seconds": 3600
      }
    },
    {
      "id": "manager_approval",
      "name": "Manager Approval",
      "step_class": "ApprovalStep",
      "config": {
        "prompt": "Manager approval required for processing",
        "channels": ["web", "email", "slack"],
        "assigned_users": ["manager@company.com", "backup-manager@company.com"],
        "timeout_seconds": 7200
      }
    },
    {
      "id": "executive_approval",
      "name": "Executive Approval",
      "step_class": "ExecutiveApprovalStep",
      "config": {
        "prompt": "Executive sign-off required",
        "channels": ["web", "email", "slack", "sms"],
        "assigned_users": ["cfo@company.com", "ceo@company.com"],
        "timeout_seconds": 14400
      }
    }
  ]
}
```

This creates a three-stage approval process where:
- Data analysts get web + email
- Managers get web + email + Slack
- Executives get all channels (urgent)
- Each has different timeout periods

