// @ts-check
const { test, expect } = require('@playwright/test');

test.describe('API Integration', () => {
  test('should have backend API running @smoke', async ({ request }) => {
    const response = await request.get('http://localhost:5555/health');
    expect(response.ok()).toBeTruthy();
    
    const body = await response.json();
    expect(body.status).toBe('healthy');
  });

  test('should return API root information', async ({ request }) => {
    const response = await request.get('http://localhost:5555/');
    expect(response.ok()).toBeTruthy();
    
    const body = await response.json();
    expect(body.name).toContain('IA Modules Showcase API');
  });

  test('should have pipelines endpoint accessible', async ({ request }) => {
    const response = await request.get('http://localhost:5555/api/pipelines');
    expect(response.ok()).toBeTruthy();
  });

  test('should have metrics endpoint accessible', async ({ request }) => {
    const response = await request.get('http://localhost:5555/api/metrics');
    expect(response.ok()).toBeTruthy();
  });
});
