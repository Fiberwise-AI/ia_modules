// @ts-check
const { test, expect } = require('@playwright/test');

test.describe('Navigation & Routing', () => {
  test('should navigate through all main pages', async ({ page }) => {
    // Start at home
    await page.goto('http://localhost:5173/');
    await page.waitForLoadState('networkidle');
    await expect(page.getByRole('heading', { name: /ia modules showcase/i })).toBeVisible();
    
    // Navigate to pipelines
    await page.getByRole('link', { name: /pipelines/i }).click();
    await page.waitForURL('**/pipelines');
    await expect(page.getByRole('heading', { name: /pipelines/i })).toBeVisible();
    
    // Navigate to metrics
    await page.getByRole('link', { name: /metrics/i }).click();
    await page.waitForURL('**/metrics');
    await expect(page.getByRole('heading', { name: /metrics/i })).toBeVisible();
    
    // Navigate to patterns
    await page.getByRole('link', { name: /patterns/i }).click();
    await page.waitForURL('**/patterns');
    await expect(page.getByRole('heading', { name: /patterns/i })).toBeVisible();
  });

  test('should handle direct URL navigation', async ({ page }) => {
    // Direct navigation to pipelines
    await page.goto('http://localhost:5173/pipelines');
    await page.waitForLoadState('networkidle');
    await expect(page.getByRole('heading', { name: /pipelines/i })).toBeVisible();
    
    // Direct navigation to metrics
    await page.goto('http://localhost:5173/metrics');
    await page.waitForLoadState('networkidle');
    await expect(page.getByRole('heading', { name: /metrics/i })).toBeVisible();
  });

  test('should maintain navigation state', async ({ page }) => {
    await page.goto('http://localhost:5173/');
    await page.waitForLoadState('networkidle');
    
    // Navigate to pipelines
    await page.getByRole('link', { name: /pipelines/i }).click();
    await page.waitForURL('**/pipelines');
    
    // Go back home
    await page.getByRole('link', { name: /home/i }).click();
    await page.waitForURL('**/');
    await expect(page.getByRole('heading', { name: /ia modules showcase/i })).toBeVisible();
  });
});
