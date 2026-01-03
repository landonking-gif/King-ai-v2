/**
 * Security Module Index
 */

export {
    validateInput,
    sanitizeString,
    containsDangerousPattern,
    sanitizePath,
    validateJSON,
    generateSecureToken,
    maskSensitiveData,
    sanitizeQuery,
} from './validation.js';

export {
    type RedisACLUser,
    DEFAULT_ACL_USERS,
    generateACLCommands,
    generateACLConfig,
} from './redis-acl.js';

export {
    redactPII,
    redactPIIFromObject,
} from './redaction.js';

export {
    type LibraryAccessPolicy,
    type LibraryAccessResult,
    resolveLibraryAccess,
} from './library-access.js';

export {
    type AuditActor,
    type AuditEvent,
    type AuditEventType,
    type AuditResource,
    logAuditEvent,
} from './audit.js';
