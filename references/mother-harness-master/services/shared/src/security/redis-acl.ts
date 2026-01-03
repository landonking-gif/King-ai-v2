/**
 * Redis ACL Configuration
 * Security configuration for Redis access control
 */

/** Redis ACL user definition */
export interface RedisACLUser {
    username: string;
    password?: string;

    // Command permissions
    commands: {
        allow: string[];
        deny: string[];
    };

    // Key patterns
    keys: string[];

    // Channel patterns
    channels: string[];

    // Status
    enabled: boolean;
}

/** Default ACL users for Mother-Harness */
export const DEFAULT_ACL_USERS: RedisACLUser[] = [
    {
        username: 'orchestrator',
        commands: {
            allow: ['get', 'set', 'del', 'keys', 'json.*', 'ft.*', 'xadd', 'xread', 'xreadgroup', 'xack', 'hget', 'hset', 'hgetall', 'hincrby', 'hincrbyfloat', 'expire', 'exists', 'ping'],
            deny: ['config', 'debug', 'shutdown', 'flushall', 'flushdb'],
        },
        keys: ['task:*', 'project:*', 'approval:*', 'model_decision:*', 'cost:*', 'budget:*', 'retry:*'],
        channels: ['notifications:*'],
        enabled: true,
    },
    {
        username: 'docling',
        commands: {
            allow: ['get', 'set', 'del', 'keys', 'json.*', 'xadd', 'xread', 'xreadgroup', 'xack', 'xgroup', 'ping'],
            deny: ['config', 'debug', 'shutdown', 'flushall', 'flushdb'],
        },
        keys: ['doc:*', 'library:*', 'docling_job:*', 'stream:docling'],
        channels: [],
        enabled: true,
    },
    {
        username: 'agents',
        commands: {
            allow: ['get', 'set', 'keys', 'json.*', 'ft.search', 'ping'],
            deny: ['config', 'debug', 'shutdown', 'flushall', 'flushdb', 'del'],
        },
        keys: ['task:*', 'project:*', 'doc:*', 'tool:*', 'role:*', 'memory:*'],
        channels: [],
        enabled: true,
    },
    {
        username: 'dashboard',
        commands: {
            allow: ['get', 'keys', 'json.get', 'ft.search', 'ping'],
            deny: ['set', 'del', 'config', 'debug', 'shutdown', 'flushall', 'flushdb'],
        },
        keys: ['task:*', 'project:*', 'approval:*', 'library:*'],
        channels: ['notifications:*'],
        enabled: true,
    },
];

/**
 * Generate Redis ACL commands for user setup
 */
export function generateACLCommands(user: RedisACLUser): string[] {
    const commands: string[] = [];

    // Reset user
    commands.push(`ACL DELUSER ${user.username}`);

    if (!user.enabled) {
        return commands;
    }

    // Build user command
    let aclCmd = `ACL SETUSER ${user.username} on`;

    // Add password if set
    if (user.password) {
        aclCmd += ` >${user.password}`;
    } else {
        aclCmd += ' nopass';
    }

    // Add allowed commands
    for (const cmd of user.commands.allow) {
        aclCmd += ` +${cmd}`;
    }

    // Add denied commands
    for (const cmd of user.commands.deny) {
        aclCmd += ` -${cmd}`;
    }

    // Add key patterns
    for (const pattern of user.keys) {
        aclCmd += ` ~${pattern}`;
    }

    // Add channel patterns
    for (const pattern of user.channels) {
        aclCmd += ` &${pattern}`;
    }

    commands.push(aclCmd);

    return commands;
}

/**
 * Generate ACL config file content
 */
export function generateACLConfig(users: RedisACLUser[]): string {
    const lines: string[] = [
        '# Mother-Harness Redis ACL Configuration',
        '# Generated automatically - do not edit manually',
        '',
    ];

    for (const user of users) {
        if (!user.enabled) continue;

        lines.push(`# User: ${user.username}`);

        let userLine = `user ${user.username} on`;

        if (user.password) {
            userLine += ` >${user.password}`;
        } else {
            userLine += ' nopass';
        }

        for (const cmd of user.commands.allow) {
            userLine += ` +${cmd}`;
        }

        for (const cmd of user.commands.deny) {
            userLine += ` -${cmd}`;
        }

        for (const pattern of user.keys) {
            userLine += ` ~${pattern}`;
        }

        lines.push(userLine);
        lines.push('');
    }

    return lines.join('\n');
}
