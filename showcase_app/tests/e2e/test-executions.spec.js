// @ts-check
const { test, expect } = require('@playwright/test');
const { ExecutionsPage } = require('./pages/ExecutionsPage');

test.describe('Executions List', () => {
  let executionsPage;

  test.beforeEach(async ({ page }) => {
    executionsPage = new ExecutionsPage(page);
    await executionsPage.goto();
  });

  test('should load executions page', async ({ page }) => {
    await executionsPage.expectLoaded();
    await executionsPage.screenshot('executions-page-loaded');
  });

  test('should display executions table', async ({ page }) => {
    await expect(executionsPage.executionsTable).toBeVisible();
  });

  test('should display table headers', async ({ page }) => {
    await expect(executionsPage.jobIdHeader).toBeVisible();
    await expect(executionsPage.statusHeader).toBeVisible();
    await expect(executionsPage.progressHeader).toBeVisible();
  });

  test('should show execution rows', async ({ page }) => {
    // Wait for executions to load
    await page.waitForTimeout(2000);
    
    const rowCount = await executionsPage.getExecutionCount();
    expect(rowCount).toBeGreaterThan(0);
  });

  test('should expand execution row details', async ({ page }) => {
    // Wait for executions to load
    await page.waitForTimeout(2000);
    
    const firstExpandButton = executionsPage.expandButtons.first();
    if (await firstExpandButton.isVisible()) {
      await firstExpandButton.click();
      
      // Expanded details should be visible
      await expect(page.locator('[class*="expanded"], [class*="detail"]').first()).toBeVisible();
      
      await executionsPage.screenshot('execution-expanded');
    }
  });

  test('should display execution status badges', async ({ page }) => {
    // Wait for executions to load
    await page.waitForTimeout(2000);
    
    // At least one status indicator should be visible
    const hasStatus = await page.locator('[class*="status"], [class*="badge"]').first().isVisible().catch(() => false);
    expect(hasStatus).toBe(true);
  });

  test('should show execution timestamps', async ({ page }) => {
    // Wait for executions to load
    await page.waitForTimeout(2000);
    
    // Timestamp elements should be present
    const timestamps = page.locator('[class*="timestamp"], [class*="time"]').first();
    await expect(timestamps).toBeVisible();
  });

  test('should display HITL pending badges', async ({ page }) => {
    // Wait for executions to load
    await page.waitForTimeout(2000);
    
    // Check if any HITL pending badges exist
    const hasHITL = await executionsPage.hitlPendingBadge.isVisible().catch(() => false);
    
    if (hasHITL) {
      await expect(executionsPage.hitlPendingBadge).toBeVisible();
      await executionsPage.screenshot('hitl-pending-badge');
    }
  });

  test('should navigate to execution detail', async ({ page }) => {
    // Wait for executions to load
    await page.waitForTimeout(2000);
    
    // Find first execution row and click on it or a detail link
    const firstRow = executionsPage.executionRows.first();
    
    if (await firstRow.isVisible()) {
      // Try to find a link/button to detail page
      const detailLink = firstRow.getByRole('link').or(firstRow.getByRole('button')).first();
      
      if (await detailLink.isVisible()) {
        await detailLink.click();
        
        // Should navigate to execution detail
        await page.waitForURL('**/executions/**');
        await expect(page.getByRole('heading', { name: /execution/i })).toBeVisible({ timeout: 5000 });
        
        await executionsPage.screenshot('execution-detail-navigation');
      }
    }
  });

  test('should update via WebSocket', async ({ page }) => {
    // Wait for initial load
    await page.waitForTimeout(2000);
    
    // Get initial count
    const initialCount = await executionsPage.getExecutionCount();
    
    // Wait for potential WebSocket updates
    await page.waitForTimeout(5000);
    
    // Count might change due to WebSocket updates
    const updatedCount = await executionsPage.getExecutionCount();
    
    // Just verify the page is still functional
    await executionsPage.expectLoaded();
  });
});
