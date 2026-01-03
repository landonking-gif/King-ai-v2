/**
 * Library Access Control
 */

export interface LibraryAccessPolicy {
    allowedLibraryIds?: string[];
    deniedLibraryIds?: string[];
    roleOverrides?: Record<string, { allow?: string[]; deny?: string[] }>;
}

export interface LibraryAccessResult {
    allowed: string[];
    denied: string[];
}

function normalizeIds(ids?: string[]): string[] {
    return (ids ?? [])
        .map(id => id.trim())
        .filter(Boolean);
}

/**
 * Resolve library access for a user based on roles and policy.
 */
export function resolveLibraryAccess(
    requestedLibraryIds: string[],
    roles: string[],
    policy: LibraryAccessPolicy = {}
): LibraryAccessResult {
    const requested = Array.from(new Set(normalizeIds(requestedLibraryIds)));

    if (roles.includes('admin')) {
        return { allowed: requested, denied: [] };
    }

    const baseAllowed = policy.allowedLibraryIds
        ? requested.filter(id => normalizeIds(policy.allowedLibraryIds).includes(id))
        : requested;

    const allowed = new Set(baseAllowed);
    const denied = new Set(normalizeIds(policy.deniedLibraryIds));

    for (const role of roles) {
        const override = policy.roleOverrides?.[role];
        if (!override) continue;
        for (const id of normalizeIds(override.allow)) {
            allowed.add(id);
            denied.delete(id);
        }
        for (const id of normalizeIds(override.deny)) {
            allowed.delete(id);
            denied.add(id);
        }
    }

    for (const id of denied) {
        allowed.delete(id);
    }

    const allowedList = requested.filter(id => allowed.has(id));
    const deniedList = requested.filter(id => !allowed.has(id));

    return { allowed: allowedList, denied: deniedList };
}
