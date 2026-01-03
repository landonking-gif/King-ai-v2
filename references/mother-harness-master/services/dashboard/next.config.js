/** @type {import('next').NextConfig} */
const nextConfig = {
    // Transpile workspace packages that ship ESM/TS
    transpilePackages: ['@mother-harness/shared'],
    eslint: {
        ignoreDuringBuilds: true,
    },
    reactStrictMode: true,
    output: 'standalone',
    // Enable experimental features for server components
    experimental: {
        serverComponentsExternalPackages: ['ioredis'],
    },
    // API rewrites to orchestrator
    async rewrites() {
        return [
            {
                source: '/api/:path*',
                destination: `${process.env.ORCHESTRATOR_URL || 'http://192.168.50.219:8002'}/api/:path*`,
            },
            {
                source: '/health',
                destination: `${process.env.ORCHESTRATOR_URL || 'http://192.168.50.219:8002'}/health`,
            },
        ];
    },
};

module.exports = nextConfig;
