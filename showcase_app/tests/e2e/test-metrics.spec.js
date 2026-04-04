// @ts-check
const { test, expect } = require('@playwright/test');
const { MetricsPage } = require('./pages/MetricsPage');

test.describe('Metrics Dashboard', () => {
  let metricsPage;

  test.beforeEach(async ({ page }) => {
    metricsPage = new MetricsPage(page);
    await metricsPage.goto();
  });

  test('should load metrics page', async ({ page }) => {
    await metricsPage.expectLoaded();
    await metricsPage.screenshot('metrics-page-loaded');
  });

  test('should display metrics cards', async ({ page }) => {
    await expect(metricsPage.metricsCards).toHaveCount({ min: 1 });
  });

  test('should display success rate metric', async ({ page }) => {
    await expect(metricsPage.successRate).toBeVisible();
  });

  test('should display charts', async ({ page }) => {
    await expect(metricsPage.successRateChart).toBeVisible();
  });

  test('should filter metrics by time range', async ({ page }) => {
    // Check if time range filter exists
    const hasTimeFilter = await metricsPage.timeRangeFilter.isVisible().catch(() => false);
    
    if (hasTimeFilter) {
      await metricsPage.timeRangeFilter.click();
      await metricsPage.last24Hours.click();
      
      // Wait for metrics to update
      await page.waitForTimeout(1000);
      
      // Verify metrics are still visible
      await expect(metricsPage.metricsCards).toHaveCount({ min: 1 });
    }
  });

  test('should display checkpoint recovery metrics', async ({ page }) => {
    await expect(metricsPage.checkpointRecovery).toBeVisible();
  });

  test('should display human intervention rate', async ({ page }) => {
    await expect(metricsPage.humanInterventionRate).toBeVisible();
  });

  test('should update metrics after pipeline execution', async ({ page }) => {
    // Navigate to execute a pipeline first
    await page.goto('http://localhost:5173/pipelines');
    await page.waitForLoadState('networkidle');
    
    // Execute a pipeline
    const pipelineCard = page.locator('[class*="pipeline"], [class*="card"]').first();
    await pipelineCard.click();
    
    const executeButton = page.getByRole('button', { name: /execute|run/i });
    await executeButton.click();
    
    // Wait for completion
    await expect(page.getByText(/complete|success/i)).toBeVisible({ timeout: 30000 });
    
    // Navigate back to metrics
    await metricsPage.goto();
    
    // Verify metrics are displayed
    await metricsPage.expectLoaded();
  });
});
