/**
 * Artillery custom processors for load testing
 */

export function randomQuery(context, events, done) {
  const queries = [
    "Research best practices for React testing",
    "Analyze API performance metrics",
    "Design a user authentication system",
    "Code a simple calculator function",
    "Review security vulnerabilities in web applications",
    "Research microservices architecture patterns",
    "Analyze database query optimization",
    "Design a caching strategy for APIs",
    "Implement rate limiting middleware",
    "Review code quality metrics"
  ];

  context.vars.randomQuery = queries[Math.floor(Math.random() * queries.length)];
  return done();
}

export function randomNumber(context, events, done) {
  context.vars.randomNum = Math.floor(Math.random() * 10000);
  return done();
}

export function logProgress(context, events, done) {
  if (context.vars.taskId) {
    console.log(`Task ${context.vars.taskId} - Status: ${context.vars.taskStatus || 'unknown'}`);
  }
  return done();
}
