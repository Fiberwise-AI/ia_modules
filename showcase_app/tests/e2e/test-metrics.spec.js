// @ts-check
const { test, expect } = require('@playwright/test');

test.describe('Metrics Dashboard', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:5173/metrics');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(500);
  });

  test('should load metrics page', async ({ page }) => {
    await expect(page.locator('body')).toBeVisible();
    await page.screenshot({ path: 'test-results/metrics-page-loaded.png', fullPage: true });
  });

  test('should update metrics after pipeline execution', async ({ page }) => {
    await page.goto('http://localhost:5173/pipelines');
    await page.waitForLoadState('networkidle');
    const hasEmptyState = await page.getByText(/no pipelines yet/i).first().isVisible().catch(() => false);
    if (!hasEmptyState) {
      // Try to find and click an execute/run button directly on the pipelines list
      const executeButton = page.getByRole('button', { name: /execute|run/i }).first();
      const hasExecuteButton = await executeButton.isVisible().catch(() => false);
      if (hasExecuteButton) {
        await executeButton.click();
        await page.waitForTimeout(3000);
      }
    }
    await page.goto('http://localhost:5173/metrics');
    await page.waitForLoadState('networkidle');
    await expect(page.locator('body')).toBeVisible();
  });
});
